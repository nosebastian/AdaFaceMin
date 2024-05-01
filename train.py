from argparse import Namespace, ArgumentParser
from adafacemin.model import FaceRecognitionModel
from adafacemin.datamodule import SampleDataModule
from adafacemin.utils import load_matching_state_dict

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch



def main(args: Namespace):
    model = FaceRecognitionModel(
        model_name=args.model_name,
        head_type=args.head_type,
        class_num=args.class_num,
        m=args.m,
        t_alpha=args.t_alpha,
        h=args.h,
        s=args.s,
    )
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            load_matching_state_dict(model, checkpoint['state_dict'])
        except KeyError as e:
            print(f"Maybe wrong model size? KeyError: {e}")
            raise e
    
    data_module = SampleDataModule(
        train_root=args.train_root,
        val_root=args.val_root,
        test_root=args.test_root,
        test_sets=args.test_sets,
        
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    
    loggers = [
        TensorBoardLogger('.')
    ]
    
    callbacks = [
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', save_last=True),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        logger=loggers,
        callbacks=callbacks,
    )
    
    if args.train:
        trainer.fit(model, data_module)
    if args.test:
        trainer.test(model, data_module)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ir_101')
    parser.add_argument('--head_type', type=str, default='arcface')
    
    parser.add_argument('--class_num', type=int, default=6)
    
    parser.add_argument('--m', type=float, default=0.4)
    parser.add_argument('--t_alpha', type=float, default=0.333)
    parser.add_argument('--h', type=float, default=64.)
    parser.add_argument('--s', type=float, default=1.0)
    parser.add_argument('--train_root', type=str, default='./data/train')
    parser.add_argument('--val_root', type=str, default='./data/val')
    parser.add_argument('--test_root', type=str, default='./data/test')
    parser.add_argument('--test_sets', type=str, nargs='+', default=['lfw','cfp_fp','cplfw','calfw','agedb_30'])#,'cfp_ff', 'vgg2_fp'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--accelerator', choices=["cpu","gpu","tpu","ipu","hpu","mps","auto"], default='gpu')
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--precision', choices=['64','64-true','32','32-true','16','16-mixed','bf16','bf16-mixed'], default='16')
    
    checkpoint_group = parser.add_mutually_exclusive_group(required=False)
    checkpoint_group.add_argument('--checkpoint', type=str, default='weights/adaface_ir101_ms1mv3.ckpt')
    
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    if not args.train and not args.test:
        parser.error("No action specified, please specify either --train or --test or both.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args: Namespace = parse_args()
    main(args)
