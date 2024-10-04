from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .resLinkModules import UpLink
from ....layers import getChannelsbyStage, computeMaxStage, BaseConv2d, StochasticDepth
from .....utils import makeDivisible, setMethod, callMethod


class FGBottleneck(nn.Module):
    def __init__(
        self,
        InChannels: int,
        HidChannels: int=None,
        Expansion: float=2.,
        Stride: int=1,
        Dilation: int=1,
        DropRate: float=0.0, # 0 would be better
        SELayer: nn.Module=None,
        ActLayer: nn.Module=None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        # Feature guide bottleneck
        if HidChannels is None:
            HidChannels = makeDivisible(InChannels * Expansion, 8)
        self.Bottleneck = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, BNorm=True, ActLayer=nn.ReLU),
            BaseConv2d(HidChannels, HidChannels, 3, Stride, dilation=Dilation, BNorm=True, ActLayer=nn.ReLU),
            SELayer(InChannels=HidChannels, **kwargs) if SELayer is not None else nn.Identity(),
            BaseConv2d(HidChannels, InChannels, 1, BNorm=True)
        )
        
        self.ActLayer = ActLayer(inplace=True) if ActLayer is not None else nn.Identity()
        self.Dropout = StochasticDepth(DropRate) if DropRate > 0 else nn.Identity()
        
        
    def forward(self, x: Tensor) -> Tensor:
        Out = self.Bottleneck(x)
        return self.ActLayer(x + self.Dropout(Out))
    

class BasicBlock(nn.Module):
    def __init__(
        self,
        InChannels: int,
        OutChannels: int,
        Stride: int=1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.Conv = nn.Sequential(
            BaseConv2d(InChannels, OutChannels, 3, Stride, BNorm=True, ActLayer=nn.ReLU),
            BaseConv2d(OutChannels, OutChannels, 1, BNorm=True, ActLayer=nn.ReLU),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.Conv(x)


class CSLayer(nn.Module):
    def __init__(
        self,
        opt,
        InChannels: int,
        OutChannels: int,
        Stride: int=1,
        NumBlocks: int=1,
        Stage: int=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.NumBlocks = NumBlocks

        Block = BasicBlock
        
        for i in range(NumBlocks + 1):
            if i < NumBlocks:
                OutChannelsC1 = InChannels if i < NumBlocks - 1 else OutChannels
                Layer = Block(InChannels, OutChannelsC1, Stride, Reparamed=opt.reparamed, **kwargs)
                Name = 'Conv%d' % (i + 1)
                setMethod(self, Name, Layer)

    def forward(self, x: Tensor) -> Tensor:
        SubFeatureTuple = [x]
        for i in range(self.NumBlocks):
            Name = 'Conv%d' % (i + 1)
            Output = callMethod(self, Name)(SubFeatureTuple[-1])
            SubFeatureTuple.append(Output)
        
        return SubFeatureTuple[-1]    


class FGLink(nn.Module):
    '''
    similar to reslink, used in feature guide
    '''
    def __init__(self, opt, ModelConfigDict, **kwargs: Any) -> None:
        super().__init__()
        self.opt = opt
        self.UpLink = []
        # self.StageList = [] # different from reslink, it use encoded feature tuple
        MaxStage = computeMaxStage(ModelConfigDict)
        
        InChannels = getChannelsbyStage(ModelConfigDict, MaxStage)
        for i, s in enumerate(reversed(range(opt.fg_start_stage, MaxStage))):
            OutChannels = getChannelsbyStage(ModelConfigDict, s)
 
            UpConv = UpLink(InChannels, OutChannels)
            setMethod(self, 'UpLink%d' % (i + 1), UpConv)
            
            if opt.fg_link == 2:
                CatLinkConv = BasicBlock(OutChannels * 2, OutChannels)
                setMethod(self, 'CatLinkConv%d' % (i + 1), CatLinkConv)
            
            InChannels = OutChannels
            # self.StageList.append(s) 

    def forwardFeature(self, FeaturesTuple: list) -> list:
        DecodeFeatureTuple = []
        DecodeFeature = FeaturesTuple[-1]
        for i, f in enumerate(reversed(FeaturesTuple[self.opt.fg_start_stage - 1:-1])):
            UpLink = callMethod(self, 'UpLink%d' % (i + 1))
            DecodeFeature = UpLink(DecodeFeature)
            if self.opt.fg_link == 1:
                DecodeFeature = DecodeFeature + f
            else:
                DecodeFeature = torch.cat([DecodeFeature, f], axis=1)
                
                CatLinkConv = callMethod(self, 'CatLinkConv%d' % (i + 1))
                DecodeFeature = CatLinkConv(DecodeFeature)
            
            DecodeFeatureTuple.append(DecodeFeature)
        return DecodeFeatureTuple
        
    def forward(self, FeaturesTuple: list) -> Tensor:
        return self.forwardFeature(FeaturesTuple)[-1]
