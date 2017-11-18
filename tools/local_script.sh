KEY_PEM_DIR=/home/hwang/My_Code/AWS/HongyiScript.pem
KEY_PEM_NAME=HongyiScript.pem
PUB_IP_ADDR="$1"
echo "Public address of master node: ${PUB_IP_ADDR}"

ssh -o "StrictHostKeyChecking no" ubuntu@${PUB_IP_ADDR}
scp -i ${KEY_PEM_DIR} ${KEY_PEM_DIR} ubuntu@${PUB_IP_ADDR}:~/.ssh
scp -i ${KEY_PEM_DIR} hosts hosts_address config ubuntu@${PUB_IP_ADDR}:~/
scp -i ${KEY_PEM_DIR} -r /home/hwang/My_Code/adversarial_coding ubuntu@${PUB_IP_ADDR}:~/
ssh -i ${KEY_PEM_DIR} ubuntu@${PUB_IP_ADDR} 'sudo apt-get update; cp adversarial_coding/tools/remote_script.sh ~/'