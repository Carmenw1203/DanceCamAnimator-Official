from args import parse_train_opt
from EditableDanceCameraVelocity import EditableDanceCameraVelocity

def train(opt):
    model = EditableDanceCameraVelocity(feature_type = opt.feature_type,
                            w_loss = opt.w_loss,
                            w_v_loss = opt.w_v_loss,
                            w_a_loss = opt.w_a_loss,
                            w_in_ba_loss = opt.w_in_ba_loss,
                            w_out_ba_loss = opt.w_out_ba_loss,)
    
    model.train_loop(opt)

if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)