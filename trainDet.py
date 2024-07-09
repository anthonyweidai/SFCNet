from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch

from lib.external import setup, registerMyD2Dataset, TrainEngine


def main(args):
    cfg = setup(args)
    
    registerMyD2Dataset()

    if not args.eval_only:
        Trainer = TrainEngine(cfg)
        Trainer.resume_or_load(resume=args.resume)
        return Trainer.train()
    else:
        Model = TrainEngine.build_model(cfg)
        DetectionCheckpointer(Model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return TrainEngine.test(cfg, Model)


if __name__ == "__main__":
    """ Modified https://github.com/facebookresearch/moco/tree/main/detection
    Available datasets: 
    voc_2012_trainval, voc_2007_trainval, voc_2007_test
    Registered dataset:
    spermtrack_2023_train, spermtrack_2023_test,
    """
    # sys.argv[1]
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )