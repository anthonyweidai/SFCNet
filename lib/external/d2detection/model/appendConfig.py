from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True
_C.INPUT.IS_ROTATE = False

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# The options for the quality of box prediction
# It can be "ctrness" (as described in FCOS paper) or "iou"
# Using "iou" here generally has ~0.4 better AP on COCO
# Note that for compatibility, we still use the term "ctrness" in the code
_C.MODEL.FCOS.BOX_QUALITY = "ctrness"

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# The normalizer of the classification loss
# The normalizer can be "fg" (normalized by the number of the foreground samples),
# "moving_fg" (normalized by the MOVING number of the foreground samples),
# or "all" (normalized by the number of all samples)
_C.MODEL.FCOS.LOSS_NORMALIZER_CLS = "fg"
_C.MODEL.FCOS.LOSS_WEIGHT_CLS = 1.0

_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = "giou"
_C.MODEL.FCOS.YIELD_PROPOSAL = False
_C.MODEL.FCOS.YIELD_BOX_FEATURES = False

# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256

# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"

# ---------------------------------------------------------------------------- #
# BAText Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BATEXT = CN()
_C.MODEL.BATEXT.VOC_SIZE = 96
_C.MODEL.BATEXT.NUM_CHARS = 25
_C.MODEL.BATEXT.POOLER_RESOLUTION = (8, 32)
_C.MODEL.BATEXT.IN_FEATURES = ["p2", "p3", "p4"]
_C.MODEL.BATEXT.POOLER_SCALES = (0.25, 0.125, 0.0625)
_C.MODEL.BATEXT.SAMPLING_RATIO = 1
_C.MODEL.BATEXT.CONV_DIM = 256
_C.MODEL.BATEXT.NUM_CONV = 2
_C.MODEL.BATEXT.RECOGNITION_LOSS = "ctc"
_C.MODEL.BATEXT.RECOGNIZER = "attn"
_C.MODEL.BATEXT.CANONICAL_SIZE = 96  # largest min_size for level 3 (stride=8)
_C.MODEL.BATEXT.USE_COORDCONV = False
_C.MODEL.BATEXT.USE_AET = False
_C.MODEL.BATEXT.EVAL_TYPE = 3 # 1: G; 2: W; 3: S
_C.MODEL.BATEXT.CUSTOM_DICT = "" # Path to the class file.

# ---------------------------------------------------------------------------- #
# BlendMask Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BLENDMASK = CN()
_C.MODEL.BLENDMASK.ATTN_SIZE = 14
_C.MODEL.BLENDMASK.TOP_INTERP = "bilinear"
_C.MODEL.BLENDMASK.BOTTOM_RESOLUTION = 56
_C.MODEL.BLENDMASK.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.BLENDMASK.POOLER_SAMPLING_RATIO = 1
_C.MODEL.BLENDMASK.POOLER_SCALES = (0.25,)
_C.MODEL.BLENDMASK.INSTANCE_LOSS_WEIGHT = 1.0
_C.MODEL.BLENDMASK.VISUALIZE = False

# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3

# ---------------------------------------------------------------------------- #
# MEInst Head
# ---------------------------------------------------------------------------- #
_C.MODEL.MEInst = CN()

# This is the number of foreground classes.
_C.MODEL.MEInst.NUM_CLASSES = 80
_C.MODEL.MEInst.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.MEInst.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.MEInst.PRIOR_PROB = 0.01
_C.MODEL.MEInst.INFERENCE_TH_TRAIN = 0.05
_C.MODEL.MEInst.INFERENCE_TH_TEST = 0.05
_C.MODEL.MEInst.NMS_TH = 0.6
_C.MODEL.MEInst.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.MEInst.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.MEInst.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.MEInst.POST_NMS_TOPK_TEST = 100
_C.MODEL.MEInst.TOP_LEVELS = 2
_C.MODEL.MEInst.NORM = "GN"  # Support GN or none
_C.MODEL.MEInst.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.MEInst.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.MEInst.LOSS_ALPHA = 0.25
_C.MODEL.MEInst.LOSS_GAMMA = 2.0
_C.MODEL.MEInst.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.MEInst.USE_RELU = True
_C.MODEL.MEInst.USE_DEFORMABLE = False
_C.MODEL.MEInst.LAST_DEFORMABLE = False
_C.MODEL.MEInst.TYPE_DEFORMABLE = "DCNv1"  # or DCNv2.

# the number of convolutions used in the cls and bbox tower
_C.MODEL.MEInst.NUM_CLS_CONVS = 4
_C.MODEL.MEInst.NUM_BOX_CONVS = 4
_C.MODEL.MEInst.NUM_SHARE_CONVS = 0
_C.MODEL.MEInst.CENTER_SAMPLE = True
_C.MODEL.MEInst.POS_RADIUS = 1.5
_C.MODEL.MEInst.LOC_LOSS_TYPE = "giou"

# ---------------------------------------------------------------------------- #
# Mask Encoding
# ---------------------------------------------------------------------------- #
# Whether to use mask branch.
_C.MODEL.MEInst.MASK_ON = True
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.MEInst.IOU_THRESHOLDS = [0.5]
_C.MODEL.MEInst.IOU_LABELS = [0, 1]
# Whether to use class_agnostic or class_specific.
_C.MODEL.MEInst.AGNOSTIC = True
# Some operations in mask encoding.
_C.MODEL.MEInst.WHITEN = True
_C.MODEL.MEInst.SIGMOID = True

# The number of convolutions used in the mask tower.
_C.MODEL.MEInst.NUM_MASK_CONVS = 4

# The dim of mask before/after mask encoding.
_C.MODEL.MEInst.DIM_MASK = 60
_C.MODEL.MEInst.MASK_SIZE = 28
# The default path for parameters of mask encoding.
_C.MODEL.MEInst.PATH_COMPONENTS = "datasets/coco/components/" \
                                   "coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60.npz"
# An indicator for encoding parameters loading during training.
_C.MODEL.MEInst.FLAG_PARAMETERS = False
# The loss for mask branch, can be mse now.
_C.MODEL.MEInst.MASK_LOSS_TYPE = "mse"

# Whether to use gcn in mask prediction.
# Large Kernel Matters -- https://arxiv.org/abs/1703.02719
_C.MODEL.MEInst.USE_GCN_IN_MASK = False
_C.MODEL.MEInst.GCN_KERNEL_SIZE = 9
# Whether to compute loss on original mask (binary mask).
_C.MODEL.MEInst.LOSS_ON_MASK = False

# ---------------------------------------------------------------------------- #
# CondInst Options
# ---------------------------------------------------------------------------- #
_C.MODEL.CONDINST = CN()

# the downsampling ratio of the final instance masks to the input image
_C.MODEL.CONDINST.MASK_OUT_STRIDE = 4
_C.MODEL.CONDINST.BOTTOM_PIXELS_REMOVED = -1

# if not -1, we only compute the mask loss for MAX_PROPOSALS random proposals PER GPU
_C.MODEL.CONDINST.MAX_PROPOSALS = -1
# if not -1, we only compute the mask loss for top `TOPK_PROPOSALS_PER_IM` proposals
# PER IMAGE in terms of their detection scores
_C.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM = -1

_C.MODEL.CONDINST.MASK_HEAD = CN()
_C.MODEL.CONDINST.MASK_HEAD.CHANNELS = 8
_C.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
_C.MODEL.CONDINST.MASK_HEAD.USE_FP16 = False
_C.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False

_C.MODEL.CONDINST.MASK_BRANCH = CN()
_C.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
_C.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 128
_C.MODEL.CONDINST.MASK_BRANCH.NORM = "BN"
_C.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 4
_C.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = False

# The options for BoxInst, which can train the instance segmentation model with box annotations only
# Please refer to the paper https://arxiv.org/abs/2012.02310
_C.MODEL.BOXINST = CN()
# Whether to enable BoxInst
_C.MODEL.BOXINST.ENABLED = False
_C.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED = 10

_C.MODEL.BOXINST.PAIRWISE = CN()
_C.MODEL.BOXINST.PAIRWISE.SIZE = 3
_C.MODEL.BOXINST.PAIRWISE.DILATION = 2
_C.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS = 10000
_C.MODEL.BOXINST.PAIRWISE.COLOR_THRESH = 0.3

# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16

# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #

_C.MODEL.BiFPN = CN()
# Names of the input feature maps to be used by BiFPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.OUT_CHANNELS = 160
_C.MODEL.BiFPN.NUM_REPEATS = 6

# Options: "" (no norm), "GN"
_C.MODEL.BiFPN.NORM = ""

# ---------------------------------------------------------------------------- #
# SOLOv2 Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SOLOV2 = CN()

# Instance hyper-parameters
_C.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.SOLOV2.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.SOLOV2.SIGMA = 0.2
# Channel size for the instance head.
_C.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
_C.MODEL.SOLOV2.INSTANCE_CHANNELS = 512
# Convolutions to use in the instance head.
_C.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
_C.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
_C.MODEL.SOLOV2.TYPE_DCN = "DCN"
_C.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
# Number of foreground classes.
_C.MODEL.SOLOV2.NUM_CLASSES = 80
_C.MODEL.SOLOV2.NUM_KERNELS = 256
_C.MODEL.SOLOV2.NORM = "GN"
_C.MODEL.SOLOV2.USE_COORD_CONV = True
_C.MODEL.SOLOV2.PRIOR_PROB = 0.01

# Mask hyper-parameters.
# Channel size for the mask tower.
_C.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
_C.MODEL.SOLOV2.MASK_CHANNELS = 128
_C.MODEL.SOLOV2.NUM_MASKS = 256

# Test cfg.
_C.MODEL.SOLOV2.NMS_PRE = 500
_C.MODEL.SOLOV2.SCORE_THR = 0.1
_C.MODEL.SOLOV2.UPDATE_THR = 0.05
_C.MODEL.SOLOV2.MASK_THR = 0.5
_C.MODEL.SOLOV2.MAX_PER_IMG = 100
# NMS type: matrix OR mask.
_C.MODEL.SOLOV2.NMS_TYPE = "matrix"
# Matrix NMS kernel type: gaussian OR linear.
_C.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
_C.MODEL.SOLOV2.NMS_SIGMA = 2

# Loss cfg.
_C.MODEL.SOLOV2.LOSS = CN()
_C.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
_C.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0


# ---------------------------------------------------------------------------- #
# FCPose Options
# ---------------------------------------------------------------------------- #

_C.MODEL.FCPOSE = CN()
_C.MODEL.FCPOSE_ON = False
_C.MODEL.FCPOSE.ATTN_LEN = 2737
_C.MODEL.FCPOSE.DYNAMIC_CHANNELS = 32
_C.MODEL.FCPOSE.MAX_PROPOSALS = 70
_C.MODEL.FCPOSE.PROPOSALS_PER_INST = 70
_C.MODEL.FCPOSE.LOSS_WEIGHT_KEYPOINT = 2.5
_C.MODEL.FCPOSE.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.FCPOSE.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.FCPOSE.GT_HEATMAP_STRIDE = 2
_C.MODEL.FCPOSE.SIGMA = 1
_C.MODEL.FCPOSE.HEATMAP_SIGMA = 1.8
_C.MODEL.FCPOSE.HEAD_HEATMAP_SIGMA = 0.01
_C.MODEL.FCPOSE.DISTANCE_NORM = 12.0
_C.MODEL.FCPOSE.LOSS_WEIGHT_DIRECTION = 9.0

_C.MODEL.FCPOSE.BASIS_MODULE = CN()
_C.MODEL.FCPOSE.BASIS_MODULE.NUM_BASES = 32
_C.MODEL.FCPOSE.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.FCPOSE.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.FCPOSE.BASIS_MODULE.NUM_CLASSES = 17
_C.MODEL.FCPOSE.BASIS_MODULE.LOSS_WEIGHT = 0.2
_C.MODEL.FCPOSE.BASIS_MODULE.BN_TYPE = "SyncBN"


# SwinT backbone
_C.MODEL.SWINT = CN()
_C.MODEL.SWINT.APE = False
_C.MODEL.SWINT.EMBED_DIM = 96
_C.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINT.MLP_RATIO = 4
_C.MODEL.SWINT.WINDOW_SIZE = 7
_C.MODEL.SWINT.DROP_PATH_RATE = 0.2
_C.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# TridentNet
_C.MODEL.TRIDENT = CN()

# Number of branches for TridentNet.
_C.MODEL.TRIDENT.NUM_BRANCH = 3
# Specify the dilations for each branch.
_C.MODEL.TRIDENT.BRANCH_DILATIONS = [1, 2, 3]
# Specify the stage for applying trident blocks. Default stage is Res4 according to the
# TridentNet paper.
_C.MODEL.TRIDENT.TRIDENT_STAGE = "res4"
# Specify the test branch index TridentNet Fast inference:
#   - use -1 to aggregate results of all branches during inference.
#   - otherwise, only using specified branch for fast inference. Recommended setting is
#     to use the middle branch.
_C.MODEL.TRIDENT.TEST_BRANCH_IDX = 1


# ---------------------------------------------------------------------------- #
# YOLO Options
# ---------------------------------------------------------------------------- #

# _C.SOLVER.OPTIMIZER = "sgd"
_C.SOLVER.LR_MULTIPLIER_OVERWRITE = []
_C.SOLVER.WEIGHT_DECAY_EMBED = 0.0

# Besides scaling default D2 configs, also scale quantization configs
_C.SOLVER.AUTO_SCALING_METHODS = [
    "default_scale_d2_configs",
    "default_scale_quantization_configs",
]

# FBNet
_C.MODEL.FBNET_V2 = CN()

_C.MODEL.FBNET_V2.ARCH = "default"
_C.MODEL.FBNET_V2.ARCH_DEF = []
# number of channels input to trunk
_C.MODEL.FBNET_V2.STEM_IN_CHANNELS = 3
_C.MODEL.FBNET_V2.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET_V2.WIDTH_DIVISOR = 1

# normalization configs
# name of norm such as "bn", "sync_bn", "gn"
_C.MODEL.FBNET_V2.NORM = "bn"
# for advanced use case that requries extra arguments, passing a list of
# dict such as [{"num_groups": 8}, {"momentum": 0.1}] (merged in given order).
# Note that string written it in .yaml will be evaluated by yacs, thus this
# node will become normal python object.
# https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L410
_C.MODEL.FBNET_V2.NORM_ARGS = []

# VT FPN
_C.MODEL.VT_FPN = CN()

_C.MODEL.VT_FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.VT_FPN.OUT_CHANNELS = 256
_C.MODEL.VT_FPN.LAYERS = 3
_C.MODEL.VT_FPN.TOKEN_LS = [16, 16, 8, 8]
_C.MODEL.VT_FPN.TOKEN_C = 1024
_C.MODEL.VT_FPN.HEADS = 16
_C.MODEL.VT_FPN.MIN_GROUP_PLANES = 64
_C.MODEL.VT_FPN.NORM = "BN"
_C.MODEL.VT_FPN.POS_HWS = []
_C.MODEL.VT_FPN.POS_N_DOWNSAMPLE = []

# SparseInst
_C.MODEL.SPARSE_INST = CN()

# parameters for inference
_C.MODEL.SPARSE_INST.CLS_THRESHOLD = 0.005
_C.MODEL.SPARSE_INST.MASK_THRESHOLD = 0.45
_C.MODEL.SPARSE_INST.MAX_DETECTIONS = 100

# [Encoder]
_C.MODEL.SPARSE_INST.ENCODER = CN()
_C.MODEL.SPARSE_INST.ENCODER.NAME = "FPNPPMEncoder"
_C.MODEL.SPARSE_INST.ENCODER.NORM = ""
_C.MODEL.SPARSE_INST.ENCODER.IN_FEATURES = ["res3", "res4", "res5"]
_C.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS = 256

# [Decoder]
_C.MODEL.SPARSE_INST.DECODER = CN()
_C.MODEL.SPARSE_INST.DECODER.NAME = "BaseIAMDecoder"
_C.MODEL.SPARSE_INST.DECODER.NUM_MASKS = 100
_C.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = 80
# kernels for mask features
_C.MODEL.SPARSE_INST.DECODER.KERNEL_DIM = 128
# upsample factor for output masks
_C.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR = 2.0
_C.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM = False
_C.MODEL.SPARSE_INST.DECODER.GROUPS = 4
# decoder.inst_branch
_C.MODEL.SPARSE_INST.DECODER.INST = CN()
_C.MODEL.SPARSE_INST.DECODER.INST.DIM = 256
_C.MODEL.SPARSE_INST.DECODER.INST.CONVS = 4
# decoder.mask_branch
_C.MODEL.SPARSE_INST.DECODER.MASK = CN()
_C.MODEL.SPARSE_INST.DECODER.MASK.DIM = 256
_C.MODEL.SPARSE_INST.DECODER.MASK.CONVS = 4

# [Loss]
_C.MODEL.SPARSE_INST.LOSS = CN()
_C.MODEL.SPARSE_INST.LOSS.NAME = "SparseInstCriterion"
_C.MODEL.SPARSE_INST.LOSS.ITEMS = ("labels", "masks")
# loss weight
_C.MODEL.SPARSE_INST.LOSS.CLASS_WEIGHT = 2.0
_C.MODEL.SPARSE_INST.LOSS.MASK_PIXEL_WEIGHT = 5.0
_C.MODEL.SPARSE_INST.LOSS.MASK_DICE_WEIGHT = 2.0
# iou-aware objectness loss weight
_C.MODEL.SPARSE_INST.LOSS.OBJECTNESS_WEIGHT = 1.0

# [Matcher]
_C.MODEL.SPARSE_INST.MATCHER = CN()
_C.MODEL.SPARSE_INST.MATCHER.NAME = "SparseInstMatcher"
_C.MODEL.SPARSE_INST.MATCHER.ALPHA = 0.8
_C.MODEL.SPARSE_INST.MATCHER.BETA = 0.2

# [Dataset mapper]
_C.MODEL.SPARSE_INST.DATASET_MAPPER = "SparseInstDatasetMapper"

# CONVNEXT
_C.MODEL.CONVNEXT = CN()

_C.MODEL.CONVNEXT.OUT_FEATURES = ["dark3", "dark4", "dark5"]
_C.MODEL.CONVNEXT.WEIGHTS = ""
_C.MODEL.CONVNEXT.DEPTH_WISE = False

_C.DATASETS.CLASS_NAMES = []

# Allowed values are "normal", "softnms-linear", "softnms-gaussian", "cluster"
_C.MODEL.NMS_TYPE = "normal"
_C.MODEL.ONNX_EXPORT = False
_C.MODEL.PADDED_VALUE = 114.0
_C.MODEL.FPN.REPEAT = 2
_C.MODEL.FPN.OUT_CHANNELS_LIST = [256, 512, 1024]
# _C.MODEL.BACKBONE.STRIDE = []
# _C.MODEL.BACKBONE.CHANNEL = []

# Add Bi-FPN support
_C.MODEL.BIFPN = CN()
_C.MODEL.BIFPN.NUM_LEVELS = 5
_C.MODEL.BIFPN.NUM_BIFPN = 6
_C.MODEL.BIFPN.NORM = "GN"
_C.MODEL.BIFPN.OUT_CHANNELS = 160
_C.MODEL.BIFPN.SEPARABLE_CONV = False

_C.MODEL.REGNETS = CN()
_C.MODEL.REGNETS.TYPE = "x"
_C.MODEL.REGNETS.OUT_FEATURES = ["s2", "s3", "s4"]

# Add Input
# _C.INPUT.INPUT_SIZE = [640, 640]  # h,w order

# Add yolo config
_C.MODEL.YOLO = CN()
_C.MODEL.YOLO.NUM_BRANCH = 3
_C.MODEL.YOLO.BRANCH_DILATIONS = [1, 2, 3]
_C.MODEL.YOLO.TEST_BRANCH_IDX = 1
_C.MODEL.YOLO.VARIANT = "yolov3"  # can be yolov5 yolov7 as well
_C.MODEL.YOLO.ANCHORS = [
    [[116, 90], [156, 198], [373, 326]],
    [[30, 61], [62, 45], [42, 119]],
    [[10, 13], [16, 30], [33, 23]],
]
_C.MODEL.YOLO.ANCHOR_MASK = []
_C.MODEL.YOLO.CLASSES = 80
_C.MODEL.YOLO.MAX_BOXES_NUM = 100
_C.MODEL.YOLO.IN_FEATURES = ["dark3", "dark4", "dark5"]
_C.MODEL.YOLO.CONF_THRESHOLD = 0.01
_C.MODEL.YOLO.NMS_THRESHOLD = 0.5
_C.MODEL.YOLO.IGNORE_THRESHOLD = 0.07
_C.MODEL.YOLO.NORMALIZE_INPUT = False

_C.MODEL.YOLO.WIDTH_MUL = 1.0
_C.MODEL.YOLO.DEPTH_MUL = 1.0

_C.MODEL.YOLO.IOU_TYPE = "ciou"  # diou or iou
_C.MODEL.YOLO.LOSS_TYPE = "v4"

_C.MODEL.YOLO.LOSS = CN()
_C.MODEL.YOLO.LOSS.LAMBDA_XY = 1.0
_C.MODEL.YOLO.LOSS.LAMBDA_WH = 1.0
_C.MODEL.YOLO.LOSS.LAMBDA_CLS = 1.0
_C.MODEL.YOLO.LOSS.LAMBDA_CONF = 1.0
_C.MODEL.YOLO.LOSS.LAMBDA_IOU = 1.1
_C.MODEL.YOLO.LOSS.USE_L1 = True
_C.MODEL.YOLO.LOSS.ANCHOR_RATIO_THRESH = 4.0
_C.MODEL.YOLO.LOSS.BUILD_TARGET_TYPE = "default"

_C.MODEL.YOLO.NECK = CN()
_C.MODEL.YOLO.NECK.TYPE = "yolov3"  # default is FPN, can be pafpn as well
_C.MODEL.YOLO.NECK.WITH_SPP = False  #

_C.MODEL.YOLO.HEAD = CN()
_C.MODEL.YOLO.HEAD.TYPE = "yolox"

_C.MODEL.YOLO.ORIEN_HEAD = CN()
_C.MODEL.YOLO.ORIEN_HEAD.UP_CHANNELS = 64

# add backbone configs
_C.MODEL.DARKNET = CN()
_C.MODEL.DARKNET.DEPTH = 53
_C.MODEL.DARKNET.WITH_CSP = True
_C.MODEL.DARKNET.RES5_DILATION = 1
_C.MODEL.DARKNET.NORM = "BN"
_C.MODEL.DARKNET.STEM_OUT_CHANNELS = 32
_C.MODEL.DARKNET.OUT_FEATURES = ["dark3", "dark4", "dark5"]
_C.MODEL.DARKNET.WEIGHTS = ""
_C.MODEL.DARKNET.DEPTH_WISE = False

# add for res2nets
_C.MODEL.RESNETS.R2TYPE = "res2net50_v1d"

# # _C.MODEL.BACKBONE = CN()
# _C.MODEL.BACKBONE.SUBTYPE = "s"
# _C.MODEL.BACKBONE.PRETRAINED = True
# _C.MODEL.BACKBONE.WEIGHTS = ""
# _C.MODEL.BACKBONE.FEATURE_INDICES = [1, 4, 10, 15]
# _C.MODEL.BACKBONE.OUT_FEATURES = ["stride8", "stride16", "stride32"]


# ---------------------------------------------------------------------------- #
# MSA options
# ---------------------------------------------------------------------------- #
_C.MODEL.MSA = CN()
# Names of the input feature maps to be used by MSA
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.MSA.IN_FEATURES = []
_C.MODEL.MSA.OUT_CHANNELS = 256

# Options: "" (no norm), "GN"
_C.MODEL.MSA.NORM = ""

# Types for fusing the MSA top-down and lateral features. Can be either "sum" or "avg"
_C.MODEL.MSA.FUSE_TYPE = "sum"


# ViT
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PatchRes = 16
_C.MODEL.VIT.DimEmb = 768
_C.MODEL.VIT.Depth = 12
_C.MODEL.VIT.NumHead = 12
_C.MODEL.VIT.AUG_SHAPE = 1024
_C.MODEL.VIT.OUT_FEATURES = ["last_feat"]

# DETR
_C.MODEL.DETR = CN()
_C.MODEL.DETR.NUM_CLASSES = 1

# For Segmentation
_C.MODEL.DETR.FROZEN_WEIGHTS = ""

# LOSS
_C.MODEL.DETR.GIOU_WEIGHT = 2.0
_C.MODEL.DETR.L1_WEIGHT = 5.0
_C.MODEL.DETR.DEEP_SUPERVISION = True
_C.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1

# TRANSFORMER
_C.MODEL.DETR.NHEADS = 8
_C.MODEL.DETR.DROPOUT = 0.1
_C.MODEL.DETR.DIM_FEEDFORWARD = 2048
_C.MODEL.DETR.ENC_LAYERS = 6
_C.MODEL.DETR.DEC_LAYERS = 6
_C.MODEL.DETR.PRE_NORM = False

_C.MODEL.DETR.HIDDEN_DIM = 256
_C.MODEL.DETR.NUM_OBJECT_QUERIES = 100

_C.SOLVER.OPTIMIZER = "ADAMW"
_C.SOLVER.BACKBONE_MULTIPLIER = 0.1


def appendCfg() -> CN:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    return _C.clone()