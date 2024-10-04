import re
import atexit
import bisect
import typing
import tabulate
import pickle as pkl
from pathlib import Path
import multiprocessing as mp
from collections import deque, defaultdict, OrderedDict

import cv2
from PIL import Image

import torch
from torch import nn

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer

from lib.model import getModel
from lib.utils import pretrainedDictManager, interpolatePosEmbed, loadModelWeight


def convertPretrain2Detectron2(ModelName, OutName):
    """Modified version of 
    https://github.com/facebookresearch/detectron2/blob/
    32570b767e1b69516c58734fe6cc46005bab2aae/tools/convert-torchvision-to-d2.py
    """
    if isinstance(ModelName, str):
        print("Converting %s to its pkl version" % ModelName)
        Model = torch.load(ModelName, map_location="cpu")
    else:
        Model = ModelName.state_dict()

    NewModel = {}
    NumList = list(range(10))
    for k in list(Model.keys()):
        Key = k.lower()
        if any(s in Key for s in ["classifier", "fc", "encoderk"]):
            continue
        
        # OldKey = Key
        if "layer" not in Key:
            Key = "stem." + Key
        for n in NumList: 
            # layer1.1.0 to layer1.0
            Key = Key.replace("layer1.1.{}".format(n), "layer1.{}".format(n))
        for n in NumList:
            Key = Key.replace("layer{}".format(n), "res{}".format(n + 1))
        for n in NumList:
            Key = Key.replace("conv{}.0".format(n), "conv{}".format(n))
        for n in NumList:
            Key = Key.replace("conv{}.bn".format(n), "conv{}.norm".format(n))
        for n in NumList:
            # for torchvision pretrained model
            Key = Key.replace("bn{}".format(n), "conv{}.norm".format(n))  
        Key = Key.replace("encoder.", "") # for barlow
        Key = Key.replace("encoderq.", "") # for moco
        Key = Key.replace("downsample.conv", "shortcut")
        Key = Key.replace("downsample.bn", "shortcut.norm")
        Key = Key.replace("conv.weight", "weight")
        
        # for torchvision pretrained model
        Key = Key.replace("downsample.0", "shortcut")
        Key = Key.replace("downsample.1", "shortcut.norm")
        
        # print(OldKey, "->", Key)
        NewModel[Key] = Model.pop(k).detach().numpy()

    Res = {"model": NewModel, "__author__": "Anthony", "matching_heuristics": True}

    with open(OutName, "wb") as f:
        pkl.dump(Res, f)
        
    if Model:
        print("Unconverted keys:", Model.keys())


def convertSegDet2Detectron2(opt, ModelName, OutName):
    Substrings = re.split("_", Path(ModelName).stem)
    Model = None
    for s in Substrings[1:]:
        opt.model_name = s
        try:
            Model = getModel(opt)
        except ValueError:
            print("%s is not a model name" % s)
        else:
            print("use %s as model" % s)
            break
    
    if Model is None:
        return
        
    Model = loadModelWeight(Model, 0, ModelName, 2)
    return convertPretrain2Detectron2(Model, OutName)


def convertYOLO2Detectron2(ModelName, OutName):
    if isinstance(ModelName, str):
        print("Converting %s to match d2 version" % ModelName)
        Model = torch.load(ModelName, map_location="cpu")
    else:
        Model = ModelName.state_dict()

    NewModel = {}
    NumList = list(range(10))
    for k in list(Model.keys()):
        Key = k.lower()
        # remove unexpected key
        if any(s in Key for s in ["classifier", "fc", "encoderk"]):
            continue
        # for n in NumList:
        elif int(re.findall(r"\d+", k)[0]) >= 11:
            continue
    
        OldKey = Key
        Key = Key.replace("model.0.", "stem.")
        for n in NumList: 
            Key = Key.replace(".cv{}.".format(n), ".layer{}.".format(n))
            
        for i in [0, 1]:
            Key = Key.replace("model.{}.".format(i + 1), "dark1.{}.".format(i))
            Key = Key.replace("model.4.{}.".format(i), "dark2.{}.".format(i + 1))
            
        for s, t in zip(list(range(6, 12, 2)), list(range(3, 6))):
            for n in NumList:
                # e.g., Key.replace("model.6.{}".format(t), "dark3.{}".format(t + 1))
                Key = Key.replace("model.%d.%d." % (s, n), "dark%d.%d." % (t, n + 1))
                
        for s, t in zip(list(range(3, 11, 2)), list(range(2, 6))):
            # e.g., Key.replace("model.3", "dark2.0")
            Key = Key.replace("model.{}.".format(s), "dark{}.0.".format(t))
            
        print(OldKey, "->", Key)
        NewModel[Key] = Model.pop(k).detach().numpy()
        
    # numpy to tensor
    for k, v in NewModel.items():
        NewModel[k] = torch.from_numpy(v).to(torch.float32)
    # dict to order dict
    NewModel = OrderedDict(NewModel.items())
    
    torch.save(NewModel, OutName)


def convertViT2Detectron2(TargetModel, ModelName, OutName):
    if isinstance(ModelName, str):
        print("Converting %s to its pkl version" % ModelName)
        Model = torch.load(ModelName, map_location="cpu")
    else:
        Model = ModelName.state_dict()
    Model = pretrainedDictManager(Model)
    Model = interpolatePosEmbed(TargetModel, Model)
    
    NewModel = {}
    for k in list(Model.keys()):
        Key = k.lower()
        if any(s in Key for s in ["classifier", "decoder"]):
            continue
        
        OldKey = Key
        Key = Key.replace("blocks.", "backbone.bottom_up.net.blocks.")
        print(OldKey, "->", Key)
        NewModel[Key] = Model.pop(k).detach().numpy()

    Res = {"model": NewModel, "__author__": "Anthony", "matching_heuristics": True}

    with open(OutName, "wb") as f:
        pkl.dump(Res, f)
    
    if Model:
        print("Unconverted keys:", Model.keys())


def parameterCount(model: nn.Module) -> typing.DefaultDict[str, int]:
    """
    Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
        number of parameters of the model.
    """
    r = defaultdict(int)
    for name, prm in model.named_parameters():
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    return r


def parameterCountTable(model: nn.Module, max_depth: int = 3) -> str:
    """
    Format the parameter count of the model (and its submodules or parameters)
    in a nice table. It looks like this:

    ::

        | name                            | #elements or shape   |
        |:--------------------------------|:---------------------|
        | model                           | 37.9M                |
        |  backbone                       |  31.5M               |
        |   backbone.fpn_lateral3         |   0.1M               |
        |    backbone.fpn_lateral3.weight |    (256, 512, 1, 1)  |
        |    backbone.fpn_lateral3.bias   |    (256,)            |
        |   backbone.fpn_output3          |   0.6M               |
        |    backbone.fpn_output3.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output3.bias    |    (256,)            |
        |   backbone.fpn_lateral4         |   0.3M               |
        |    backbone.fpn_lateral4.weight |    (256, 1024, 1, 1) |
        |    backbone.fpn_lateral4.bias   |    (256,)            |
        |   backbone.fpn_output4          |   0.6M               |
        |    backbone.fpn_output4.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output4.bias    |    (256,)            |
        |   backbone.fpn_lateral5         |   0.5M               |
        |    backbone.fpn_lateral5.weight |    (256, 2048, 1, 1) |
        |    backbone.fpn_lateral5.bias   |    (256,)            |
        |   backbone.fpn_output5          |   0.6M               |
        |    backbone.fpn_output5.weight  |    (256, 256, 3, 3)  |
        |    backbone.fpn_output5.bias    |    (256,)            |
        |   backbone.top_block            |   5.3M               |
        |    backbone.top_block.p6        |    4.7M              |
        |    backbone.top_block.p7        |    0.6M              |
        |   backbone.bottom_up            |   23.5M              |
        |    backbone.bottom_up.stem      |    9.4K              |
        |    backbone.bottom_up.res2      |    0.2M              |
        |    backbone.bottom_up.res3      |    1.2M              |
        |    backbone.bottom_up.res4      |    7.1M              |
        |    backbone.bottom_up.res5      |    14.9M             |
        |    ......                       |    .....             |

    Args:
        model: a torch module
        max_depth (int): maximum depth to recursively print submodules or
            parameters

    Returns:
        str: the table to be printed
    """
    count: typing.DefaultDict[str, int] = parameterCount(model)
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    param_shape: typing.Dict[str, typing.Tuple] = {
        k: tuple(v.shape) for k, v in model.named_parameters()
    }

    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    table: typing.List[typing.Tuple] = []

    def format_size(x: int) -> str:
        if x > 1e8:
            return "{:.2f}G".format(x / 1e9)
        if x > 1e5:
            return "{:.2f}M".format(x / 1e6)
        if x > 1e2:
            return "{:.2f}K".format(x / 1e3)
        return str(x)

    def fill(lvl: int, prefix: str) -> None:
        if lvl >= max_depth:
            return
        for name, v in count.items():
            if name.count(".") == lvl and name.startswith(prefix):
                indent = " " * (lvl + 1)
                if name in param_shape:
                    table.append((indent + name, indent + str(param_shape[name])))
                else:
                    table.append((indent + name, indent + format_size(v)))
                    fill(lvl + 1, name + ".")

    table.append(("model", format_size(count.pop(""))))
    fill(0, "")

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(
        table, headers=["name", "#elements or shape"], tablefmt="pipe"
    )
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab


# predictor
# Copyright (c) Facebook, Inc. and its affiliates.
# https://github.com/facebookresearch/detectron2/blob/main/demo/predictor.py
class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            if "cpu" not in cfg.MODEL.DEVICE:
                # use gpu as default
                cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
    

class VisualizationDemo:
    def __init__(
        self, 
        cfg, 
        NumGPU: int=1, 
        instance_mode=ColorMode.IMAGE, 
        parallel=False, 
        BBoxAcc=False
    ):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel and NumGPU > 1:
            self.predictor = AsyncPredictor(cfg, num_gpus=NumGPU)
        else:
            self.predictor = DefaultPredictor(cfg)
        
        self.BBoxAcc = BBoxAcc

    def run_on_image(self, image):
        """
        https://github.com/facebookresearch/detectron2/issues/1519#issuecomment-658961572
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            Predictions (dict): the output of the model.
            VisOutput (VisImage): the visualized image output.
        """
        VisOutput = None
        Predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1] 
        VisDrawer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        
        if "panoptic_seg" in Predictions:
            panoptic_seg, segments_info = Predictions["panoptic_seg"]
            VisOutput = VisDrawer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in Predictions:
                VisOutput = VisDrawer.draw_sem_seg(
                    Predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in Predictions:
                instances = Predictions["instances"].to(self.cpu_device)
                if self.BBoxAcc:
                    # output with prediction accuracy
                    VisOutput = VisDrawer.draw_instance_predictions(predictions=instances)
                else:
                    # output with only prediction bbox
                    # https://stackoverflow.com/a/64890075/15329637
                    if self.instance_mode.value == 0:
                        NumInstances = len(instances.pred_boxes)
                        AssignedColors = [random_color(rgb=True, maximum=1) for _ in range(NumInstances)]
                    elif self.instance_mode.value == 1:
                        AssignedColors = [[c / 255 for c in self.metadata.get("thing_colors")[0]]]
                    else:
                        # grey
                        AssignedColors = [[0, 0, 0]]
                    
                    for i, b in enumerate(instances.pred_boxes):
                        VisDrawer.draw_box(
                            b, edge_color=AssignedColors[i] 
                            if len(AssignedColors) > 1 else AssignedColors[0]
                        )
                    VisOutput = VisDrawer.get_output()
                    VisOutput = VisOutput.get_image()
                    VisOutput = Image.fromarray(VisOutput)

        return Predictions, VisOutput

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes Predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, Predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in Predictions:
                panoptic_seg, segments_info = Predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in Predictions:
                Predictions = Predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, Predictions)
            elif "sem_seg" in Predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, Predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    Predictions = self.predictor.get()
                    yield process_predictions(frame, Predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                Predictions = self.predictor.get()
                yield process_predictions(frame, Predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))

