python train.py --dataset=fashion_mnist --epoch=25 --adversarial_loss_mode=gan

python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=dragan

python train.py --dataset=anime --epoch=200 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=5

python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan
python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=dragan
python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=lp --gradient_penalty_sample_mode=dragan
python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=lsgan
python train.py --dataset=celeba --epoch=50 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=5
python train.py --dataset=celeba --epoch=50 --adversarial_loss_mode=wgan --gradient_penalty_mode=lp --gradient_penalty_sample_mode=line --n_d=5

python train.py --dataset_name=pose10 --adversarial_loss_mode=gan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=dragan
# 10张图片1张1张的进，有些动作没有训练出来

python train.py --dataset_name=pose10 --adversarial_loss_mode=gan
# 训练比较清晰，但是只是一个动作无多样性

python train.py --dataset_name=pose10 --adversarial_loss_mode=gan --gradient_penalty_mode=lp --gradient_penalty_sample_mode=dragan
# 会出现两个人，但是两个人这种颜色比较浅(可以考虑maxpool)

python train.py --dataset_name=pose10 --epochs=1000 --adversarial_loss_mode=lsgan
#训练在5000多次时较稳定，但只有一个动作,无多样性

python train.py --dataset_name=pose10 --epochs=1000 --adversarial_loss_mode=wgan
#训练崩溃

python train.py --dataset=pose10 --epoch=1000 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=5
#总样本只有n_d=1的1/4，总体比较模糊

python train.py --dataset=pose10 --epoch=1000 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=2
#动作比较全！，部分动作会演化为双人,双人效果也较多

python train.py --dataset=pose10 --epoch=2000 --adversarial_loss_mode=wgan --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=dragan --n_d=2
#训练收敛比较慢，中间出现过崩溃，后期可以生成但比较模糊，介于模糊和崩溃之间