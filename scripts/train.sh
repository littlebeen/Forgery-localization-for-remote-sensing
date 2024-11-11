# only train content-based prior model

python src/main.py --data_train Vaihingengt --data_train_dir fakeV --model restore

# only train M-FLnet (without forgery detection, optional)

#python src/main.py --data_train Vaihingen --data_train_dir fakeV --model mflnet

# train FLDCF

python src/main.py --data_train Vaihingen --data_train_dir fakeV --model fldcf


# test

python src/test.py --data_train Vaihingen --data_train_dir fakeV --model fldcf --pre_train model_fakeV.pt