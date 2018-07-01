mpirun -n 8 --hostfile hosts_address \
python distributed_nn.py \
--lr=0.01 \
--momentum=0.9 \
--network=FC \
--dataset=MNIST \
--batch-size=4 \
--comm-type=Bcast \
--mode=maj_vote \
--approach=cyclic \
--eval-freq=2000 \
--err-mode=constant \
--adversarial=-100 \
--epochs=50 \
--max-steps=1000000 \
--worker-fail=2 \
--group-size=3 \
--compress-grad=compress \
--checkpoint-step=0 \
--train-dir=/home/ubuntu/