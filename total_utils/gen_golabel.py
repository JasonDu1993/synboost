# -*- coding: utf-8 -*-
# @Time    : 2021/9/29 14:45
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : gen_golabel.py
# @Software: PyCharm
import os


def get_path_len(path):
    if path.endswith("/"):
        l = len(path)
    else:
        l = len(path) + 1
    return l


def traverse(src_dir, exts=None):
    if exts is None:
        exts = ['.jpg', '.png', '.bmp']
    labels = {}
    result = []
    cnt = -1
    for root, dirs, files in sorted(os.walk(src_dir)):
        if len(files) > 0:
            class_name = root[get_path_len(src_dir):]
            if class_name not in labels:
                cnt += 1
                labels[class_name] = str(cnt)
            for f in sorted(files):
                ext = os.path.splitext(f)[-1].lower()
                if ext in exts:
                    result.append([os.path.join(root, f), str(cnt)])
    return result


def gen_golabel_from_txt(input_path, save_path):
    cnt = 0
    if os.path.isdir(input_path):
        result = traverse(input_path)
        with open(save_path, "w", encoding="utf-8") as fw:
            for lines in result:
                img_path, class_name = lines
                color = "1"
                new_line = img_path + " " + class_name + " " + color + "\n"
                fw.write(new_line)
    else:
        with open(input_path, "r", encoding="utf-8") as fr:
            with open(save_path, "w", encoding="utf-8") as fw:
                for line in fr.readlines():
                    cnt += 1
                    # if cnt % 1000 > 10:
                    #     continue
                    line_sp = line.strip().split(" ")
                    line_sp.insert(2, "1")
                    print(line_sp)
                    fw.write(" ".join(line_sp[:3]) + "\n")


def get_docker_cmd(root_dir, port):
    port = str(port)
    name = "go" + port
    cmd = "sudo docker run --rm --name " + name + " -v " + root_dir + ":/zd " + \
          "-v /cloudgpfs/workspace/zhoudu/zhoudu:/zhoudu " + \
          "-v /cloudgpfs/workspace/zhoudu/zhoudu_data:/dataset " + \
          "-v /cloudgpfs/workspace/zhouyafei:/zhouyafei " + \
          "-p " + port + ":80 artifact.cloudwalk.work/rd_docker_dev/label-app/go 10.128.8.141:" + port + " /zd"
    print(cmd)
    cmd2 = "sudo docker rm -f " + name
    print(cmd2)


if __name__ == '__main__':
    src_dir = "/dataset/dataset/ssd/gesture/common_scenario_mobile_201021/ges30_cloudwalk"
    dataset = "common_scenario_ges30_cloudwalk"
    # traverse(src_dir)
    server_dir = "/cloudgpfs/workspace/zhoudu/zhoudu/golabel"
    docker_dir = "/zhoudu/golabel"
    task = "demo"
    port = 50020
    root_dir = os.path.join(server_dir, task)
    save_path = os.path.join(docker_dir, task, "cluster_rst", dataset + ".txt")
    print("save path: {}".format(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gen_golabel_from_txt(src_dir, save_path)
    get_docker_cmd(root_dir, port)
