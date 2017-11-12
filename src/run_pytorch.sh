mpirun -n 5 \
python distributed_nn.py \
--lr=0.01 \
--network=FC \
--dataset=MNIST \
--batch-size=128 \
--comm-type=Bcast \
--num-aggregate=4 \
--mode=normal \
--kill-threshold=6.8 \
--eval-freq=200 \
--epochs=50 \
--max-steps=1000000 \
--train-dir=/home/ubuntu/