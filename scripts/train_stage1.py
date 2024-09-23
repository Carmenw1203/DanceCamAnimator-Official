from args import parse_train_stage1_opt
from MusicDance2Keyframe import MusicDance2Keyframe

def train(opt):
    model = MusicDance2Keyframe(feature_type = opt.feature_type,
                            w_positive_loss=opt.w_positive_loss,
                            w_negative_loss=opt.w_negative_loss,)
    
    model.train_loop(opt)

if __name__ == "__main__":
    opt = parse_train_stage1_opt()
    train(opt)