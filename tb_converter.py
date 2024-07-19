import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)

args = parser.parse_args()

path = args.path

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def convertPSNRtoMSE(row):
    psnr = row['value'] * -1
    mse = (255/(10**(psnr/20)))**2
    return mse

# change this path
df=tflog2pandas(path)

df = df.drop(df[df.step > 300000].index)

loss = df[df.metric.str.contains('l_pix')]
loss['mse'] = loss.apply(convertPSNRtoMSE, axis=1)

fig, ax = plt.subplots()
plt.plot(loss.step, loss.value)
plt.title("Train Loss")
plt.ylabel("Loss")
plt.xlabel("Iterations")
plt.savefig('loss.png')

metric_psnr = df[df.metric.str.contains('psnr')]
fig, ax = plt.subplots()
plt.plot(metric_psnr.step, metric_psnr.value, color="r")
plt.title("Validation Accuracy")
plt.ylabel("PSNR Accuracy")
plt.xlabel("Iterations")
plt.savefig('psnr_validation_accuracy.png')

metric_ssim = df[df.metric.str.contains('ssim')]
fig, ax = plt.subplots()
plt.plot(metric_ssim.step, metric_ssim.value, color="c")
plt.title("Validation Accuracy")
plt.ylabel("SSIM Accuracy")
plt.xlabel("Iterations")
plt.savefig('ssim_validation_accuracy.png')

metric_mse = df[df.metric.str.contains('mse')]
fig, ax = plt.subplots()
plt.plot(metric_mse.step, metric_mse.value, color="r")
plt.title("Validation Accuracy")
plt.ylabel("MSE Accuracy")
plt.xlabel("Iterations")
plt.savefig('mse_validation_accuracy.png')