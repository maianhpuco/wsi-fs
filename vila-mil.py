from transformers import T5Tokenizer, T5ForConditionalGeneration
class ViLa_MIL_Model(nn.Module):
    def __init__(self, config, num_classes=3):
        super(ViLa_MIL_Model, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

        clip_model, _ = clip.load("RN50", device="cpu")
        self.prompt_learner = PromptLearner(config.text_prompt, clip_model.float())
        self.text_encoder = TextEncoder(clip_model.float())

        self.norm = nn.LayerNorm(config.input_size)
        self.cross_attention_1 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)
        self.cross_attention_2 = MultiheadAttention(embed_dim=config.input_size, num_heads=1)

        self.learnable_image_center = nn.Parameter(torch.Tensor(*[config.prototype_number, 1, config.input_size]))
        trunc_normal_(self.learnable_image_center, std=.02)

        # Add T5 for explanation generation
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.t5_generator = T5ForConditionalGeneration.from_pretrained("t5-small")

    def forward(self, x_s, coord_s, x_l, coords_l, label):
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        M = x_s.float()
        compents, _ = self.cross_attention_1(self.learnable_image_center, M, M)
        compents = self.norm(compents + self.learnable_image_center)

        M_high = x_l.float()
        compents_high, _ = self.cross_attention_1(self.learnable_image_center, M_high, M_high)
        compents_high = self.norm(compents_high + self.learnable_image_center)

        H = compents.squeeze().float()
        A = F.softmax(self.attention_weights(self.attention_V(H) * self.attention_U(H)).transpose(1, 0), dim=1)
        image_features_low = torch.mm(A, H)

        H_high = compents_high.squeeze().float()
        A_high = F.softmax(self.attention_weights(self.attention_V(H_high) * self.attention_U(H_high)).transpose(1, 0), dim=1)
        image_features_high = torch.mm(A_high, H_high)

        text_features_low = text_features[:self.num_classes]
        image_context = torch.cat((compents.squeeze(), M), dim=0)
        text_context_features, _ = self.cross_attention_2(text_features_low.unsqueeze(1), image_context, image_context)
        text_features_low = text_context_features.squeeze() + text_features_low

        text_features_high = text_features[self.num_classes:]
        image_context_high = torch.cat((compents_high.squeeze(), M_high), dim=0)
        text_context_features_high, _ = self.cross_attention_2(text_features_high.unsqueeze(1), image_context_high, image_context_high)
        text_features_high = text_context_features_high.squeeze() + text_features_high

        logits_low = image_features_low @ text_features_low.T.cuda()
        logits_high = image_features_high @ text_features_high.T.cuda()
        logits = logits_low + logits_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        # --- Generate Explanation ---
        classnames = self.prompt_learner.tokenized_prompts.cpu().numpy()
        label_names = [self.t5_tokenizer.decode(c, skip_special_tokens=True) for c in classnames]
        batch_size = Y_hat.shape[0]
        text_inputs = [f"Explain why this slide is classified as {label_names[Y_hat[i].item()]}" for i in range(batch_size)]

        t5_input = self.t5_tokenizer(text_inputs, padding=True, truncation=True, return_tensors="pt").to(logits.device)
        decoder_output = self.t5_generator.generate(**t5_input, max_length=64)
        explanations = self.t5_tokenizer.batch_decode(decoder_output, skip_special_tokens=True)

        return Y_prob, Y_hat, loss, explanations
 