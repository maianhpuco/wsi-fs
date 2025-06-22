from tensorboard.backend.event_processing import event_accumulator
import os

# Path to TensorBoard logs
log_dir = 'processing_tcga_256/top_tcga_renal_result/runs_TOP/20250621_231442_TOP_TCGA_Renal_Fold5_Seed42_Bs1_lrTB0.001_lrIB0.001_-1Shot_bagLevelNCTX16_instLevelNCTX16_AllCTXtrainableFalse_CSCTrue_poolingStrtegylearnablePrompt_NegBagProb0.0_NegBagProP1.0_pDropOut0.5_pDropOutBag0.5_weightLossA0.0'
run_folder = sorted(os.listdir(log_dir))[-1]
ea = event_accumulator.EventAccumulator(os.path.join(log_dir, run_folder))
ea.Reload()

# print("Available scalar tags:", ea.Tags()['scalars'])


tags = ea.Tags()['scalars']

# Print final value for each tag
print(f"\n📊 Final Evaluation Metrics from: {run_folder}\n")
for tag in tags:
    try:
        events = ea.Scalars(tag)
        if events:
            print(f"{tag}: {events[-1].value:.4f}")
        else:
            print(f"{tag}: No data")
    except KeyError:
        print(f"{tag}: Not found")
