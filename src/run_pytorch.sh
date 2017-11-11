mpirun -n 3 --hostfile hosts_address \
python distributed_nn.py \
--lr=0.01 \
--network=FC \
--dataset=MNIST \
--batch-size=128 \
--comm-type=Bcast \
--num-aggregate=5 \
--mode=normal \
--kill-threshold=6.8 \
--eval-freq=200 \
--train-dir=/home/ubuntu/MPI_shared/