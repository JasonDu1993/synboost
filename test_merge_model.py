import argparse
import yaml
import torch.backends.cudnn as cudnn
import torch
from PIL import Image
import numpy as np
import os
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.utils.data
from time import time
import time as t

from image_dissimilarity.util import trainer_util, metrics
from build_model import RoadAnomalyDetector, colorize_mask
from image_dissimilarity.data.image_dataset import ImageDataset
from total_utils.seg_to_rect import postprocessing
from total_utils.draw_box_util import draw_total_box

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="image_dissimilarity/configs/test/fs_lost_found_configuration.yaml",
                    help="Path to the config file.")
parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
parser.add_argument("--store_results", type=str,
                    default="../road_obstacles/fs_lost_found_mergemodel_h1024w2048_icnet",
                    help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
with open(opts.config, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
config["store_results"] = opts.store_results
# get experiment information
exp_name = config["experiment_name"]
save_fdr = config["save_folder"]
epoch = config["which_epoch"]
store_fdr = config["store_results"]
store_fdr_exp = os.path.join(config["store_results"], exp_name)
ensemble = config["ensemble"]

# Save folders
semantic_path = os.path.join(store_fdr, "semantic")
anomaly_path = os.path.join(store_fdr, "anomaly")
synthesis_path = os.path.join(store_fdr, "synthesis")
entropy_path = os.path.join(store_fdr, "entropy")
distance_path = os.path.join(store_fdr, "distance")
perceptual_diff_path = os.path.join(store_fdr, "perceptual_diff")
box_path = os.path.join(store_fdr, "box")

os.makedirs(semantic_path, exist_ok=True)
os.makedirs(anomaly_path, exist_ok=True)
os.makedirs(synthesis_path, exist_ok=True)
os.makedirs(entropy_path, exist_ok=True)
os.makedirs(distance_path, exist_ok=True)
os.makedirs(perceptual_diff_path, exist_ok=True)
os.makedirs(box_path, exist_ok=True)

if not os.path.isdir(store_fdr):
    os.makedirs(store_fdr, exist_ok=True)

if not os.path.isdir(store_fdr_exp):
    os.makedirs(
        store_fdr_exp,
        exist_ok=True,
    )

if not os.path.isdir(os.path.join(store_fdr_exp, "pred")):
    os.makedirs(os.path.join(store_fdr_exp, "label"), exist_ok=True)
    os.makedirs(os.path.join(store_fdr_exp, "pred"), exist_ok=True)
    os.makedirs(os.path.join(store_fdr_exp, "soft"), exist_ok=True)

# Activate GPUs
config["gpu_ids"] = opts.gpu_ids
gpu_info = trainer_util.activate_gpus(config)

# checks if we are using prior images
prior = config["model"]["prior"]
# Get data loaders
cfg_test_loader = config["test_dataloader"]
# adds logic to dataloaders (avoid repetition in config file)
cfg_test_loader["dataset_args"]["prior"] = prior
input_shape = [3, 1024, 2048]  # c, h, w
cfg_test_loader["dataset_args"]["input_shape"] = input_shape

dataset = ImageDataset(**cfg_test_loader["dataset_args"])
test_loader = torch.utils.data.DataLoader(dataset, **cfg_test_loader["dataloader_args"])

# get model
vis = False
verbose = False
detector = RoadAnomalyDetector(True, input_shape, vis=vis, verbose=verbose)
detector.to(gpu_info["device"])
detector.eval()

softmax = torch.nn.Softmax(dim=1)

# create memory locations for results to save time while running the code
dataset = cfg_test_loader["dataset_args"]
h = int((dataset["crop_size"] / dataset["aspect_ratio"]))
w = int(dataset["crop_size"])
flat_pred = np.zeros(w * h * len(test_loader), dtype="float32")
flat_labels = np.zeros(w * h * len(test_loader), dtype="float32")

with torch.no_grad():
    for i, data_i in enumerate(tqdm(test_loader)):
        t0 = time()
        img = data_i["original"].cuda()
        label = data_i["label"].cuda()
        img_path = data_i["original_path"][0]
        basename = os.path.basename(img_path).replace(".jpg", ".png")
        image = cv2.imread(img_path)
        image_og_h, image_og_w, _ = image.shape
        if vis:
            t1 = time()
            prediction, anomaly_score, outs = detector(img)
            # prediction = prediction.cpu().numpy()
            # anomaly_score = anomaly_score.cpu().numpy()
            # result = postprocessing(prediction, anomaly_score)
            # print("spend total {} s".format(time() - t1))
            # draw_total_box(img_path, result[0], debug=True)

            # 可视化diss_pred
            diss_pred_img = anomaly_score[0].cpu().numpy().astype(np.uint8)
            diss_pred = Image.fromarray(diss_pred_img).resize((image_og_w, image_og_h))
            # np.save("diss_pred_resize.npy", np.array(diss_pred))
            # 可视化分割图
            seg_final_img = prediction[0].cpu().numpy().astype(np.uint8)
            semantic = Image.fromarray(seg_final_img)
            seg_img = semantic.resize((image_og_w, image_og_h))
            # np.save("seg_img.npy", np.array(seg_img))
            # 可视化合成图
            generated = outs["synthesis"]
            image_numpy = (np.transpose(generated[0].squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0 * 255
            synthesis_final_img = Image.fromarray(image_numpy.astype(np.uint8))
            synthesis = synthesis_final_img.resize((image_og_w, image_og_h))
            # 可视化entropy
            entropy = outs["softmax_entropy"]
            entropy = entropy[0].cpu().numpy()
            entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
            entropy = entropy_img.resize((image_og_w, image_og_h))
            # 可视化perceptual_diff
            perceptual_diff = outs["perceptual_diff"]
            perceptual_diff1 = perceptual_diff[0].squeeze().cpu().numpy()
            perceptual_diff = Image.fromarray(perceptual_diff1.astype(np.uint8))
            perceptual_diff = perceptual_diff.resize((image_og_w, image_og_h))
            # 可视化距离
            distance = outs["softmax_distance"]
            distance = distance[0].squeeze(0).cpu().numpy()
            distance_img = Image.fromarray(distance.astype(np.uint8).squeeze())
            distance = distance_img.resize((image_og_w, image_og_h))

            result = postprocessing(np.array(seg_img)[None, :], np.array(diss_pred)[None, :], image_og_h, image_og_w)
            img_box = draw_total_box(np.array(image), result[0], debug=True)
            results = {"anomaly_map": diss_pred, "segmentation": seg_img, "synthesis": synthesis,
                       "softmax_entropy": entropy, "perceptual_diff": perceptual_diff, "softmax_distance": distance,
                       "box": img_box}

            anomaly_map = results["anomaly_map"].convert("RGB")
            anomaly_map = Image.fromarray(
                np.concatenate([np.array(image)[:, :, ::-1], np.array(anomaly_map)], axis=1)
            )
            anomaly_map.save(os.path.join(anomaly_path, basename))

            semantic_map = colorize_mask(np.array(results["segmentation"]))
            semantic_map.save(os.path.join(semantic_path, basename))

            synthesis = results["synthesis"]
            synthesis.save(os.path.join(synthesis_path, basename))

            softmax_entropy = results["softmax_entropy"].convert("RGB")
            softmax_entropy.save(os.path.join(entropy_path, basename))

            softmax_distance = results["softmax_distance"].convert("RGB")
            softmax_distance.save(os.path.join(distance_path, basename))

            perceptual_diff = results["perceptual_diff"].convert("RGB")
            perceptual_diff.save(os.path.join(perceptual_diff_path, basename))

            img_box = results["box"]
            cv2.imwrite(os.path.join(box_path, basename), img_box)
        else:
            prediction, anomaly_score = detector(img)
            # print("spend dect {} s".format(time() - t0))
            if vis:
                # 可视化diss_pred
                diss_pred_img = anomaly_score[0].cpu().numpy()

                obs_h, obs_w = diss_pred_img.shape
                # 可视化分割图
                seg_final_img = prediction[0].cpu().numpy().astype(np.uint8)
                seg_img = cv2.resize(seg_final_img, (obs_w, obs_h))

                result = postprocessing(np.array(seg_img)[None, :], np.array(diss_pred_img)[None, :], image_og_h,
                                        image_og_w)
                img_box = draw_total_box(np.array(image), result[0], debug=True)

                draw_total_box(img_path, result[0], debug=True)
            # break
        soft_pred = anomaly_score / 255

        flat_pred[i * w * h:i * w * h +
                            w * h] = torch.flatten(soft_pred).detach().cpu().numpy()
        flat_labels[i * w * h:i * w * h +
                              w * h] = torch.flatten(label).detach().cpu().numpy()
        # Save results

        label_tensor = label * 1

        file_name = os.path.basename(data_i["original_path"][0])
        label_img = Image.fromarray(
            label_tensor.squeeze().cpu().numpy().astype(np.uint8))
        soft_img = Image.fromarray(
            (soft_pred.squeeze().cpu().numpy() * 255).astype(np.uint8))
        soft_img.save(os.path.join(store_fdr_exp, "soft", file_name))
        label_img.save(os.path.join(store_fdr_exp, "label", file_name))

print("Calculating metric scores")
if config["test_dataloader"]["dataset_args"]["roi"]:
    invalid_indices = np.argwhere(flat_labels == 255)
    flat_labels = np.delete(flat_labels, invalid_indices)
    flat_pred = np.delete(flat_pred, invalid_indices)

results = metrics.get_metrics(flat_labels, flat_pred)

print("roc_auc_score : " + str(results["auroc"]))
print("mAP: " + str(results["AP"]))
print("FPR@95%TPR : " + str(results["FPR@95%TPR"]))

if config["visualize"]:
    plt.figure()
    lw = 2
    plt.plot(results["fpr"],
             results["tpr"],
             color="darkorange",
             lw=lw,
             label="ROC curve (area = %0.2f)" % results["auroc"])
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    path = os.path.join(store_fdr_exp, "roc_curve.png")
    print("roc img save into: {}".format(path))
    plt.show()
    plt.savefig(path)
