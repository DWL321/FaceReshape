from yacs.config import CfgNode as CN

trainer_config = CN()
trainer_config.gpu_ids = [0]
trainer_config.output_dir = '/dellnas/users/guoyudong/FaceTracking/IdEmotion3D/exps/exp_0'
trainer_config.log_dir = 'log'
trainer_config.batch_size = 25
trainer_config.ckpt_path = 'model_37000.tar'
trainer_config.num_workers = 10
trainer_config.epoch_num = 100
trainer_config.log_steps = 100
trainer_config.vis_steps = 1000
trainer_config.checkpoint_steps = 1000
trainer_config.lr_update_step = 100000

trainer_config.losses_wts = CN()
trainer_config.losses_wts['col'] = 10.
# trainer_config.losses_wts['perc'] = 5.
trainer_config.losses_wts['lms'] = .5
trainer_config.losses_wts['iris'] = 1.
trainer_config.losses_wts['id'] = 1.
# trainer_config.losses_wts['emo'] = 1e-2
trainer_config.losses_wts['exp'] = 3e-2
trainer_config.losses_wts['tex'] = 5e-2

trainer_config.vid_path = '/dellnas/users/guoyudong/dataset/vids/zxx.mp4'

pipeline_config = CN()
pipeline_config.render_size = 224
pipeline_config.lr_encoder = 1e-5
pipeline_config.lr_mapper = 1e-5
pipeline_config.train_exp = True
pipeline_config.train_tex = True
pipeline_config.train_poselit = True
pipeline_config.train_shape = False
pipeline_config.with_emo = False

dataset_config = CN()
dataset_config.img_size = 400
dataset_config.list_path = ['/dellnas/users/guoyudong/FaceTracking/iris_list_valid.txt']