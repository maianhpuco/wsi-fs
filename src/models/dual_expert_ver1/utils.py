def select_patches(wsi_patches, clip_model, k_shot_patches, k_shot_labels, M=1000, K_cluster=50):
    # Tissue detection
    tissue_mask = otsu_threshold(wsi_patches)  # Placeholder
    tissue_patches = wsi_patches[tissue_mask]
    
    # Unsupervised clustering
    with torch.no_grad():
        tissue_features = clip_model.encode_image(tissue_patches)  # [N, 512]
    clusters = KMeans(n_clusters=K_cluster).fit(tissue_features.numpy())
    M1_patches = select_centroids_and_outliers(clusters, tissue_patches, M1=500)
    
    # Few-shot guided selection
    classifier = LogisticRegression().fit(k_shot_patches.numpy(), k_shot_labels.numpy())
    scores = classifier.predict_proba(tissue_features.numpy())
    M2_patches = tissue_patches[scores.argsort()[-500:]]
    
    return torch.cat([M1_patches, M2_patches], dim=0)  # [1000, C, H, W]