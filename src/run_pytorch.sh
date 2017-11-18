mpirun -n 5 \
python distributed_nn.py \
--lr=0.01 \
--network=FC \
--dataset=MNIST \
--batch-size=128 \
--comm-type=Bcast \
--num-aggregate=4 \
--mode=geometric_median \
--eval-freq=200 \
--epochs=50 \
--adversarial=-100 \
--max-steps=1000000 \
--worker-fail=4 \
--train-dir=/home/ubuntu/