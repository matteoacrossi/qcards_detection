DIR=$HOME/qcards/tf_api
export PATH=$DIR/protoc_3.10/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$DIR/models/research:$DIR/models/research/slim

# For Tensorflow to use correct GPU
export CUDA_VISIBLE_DEVICES='0'
