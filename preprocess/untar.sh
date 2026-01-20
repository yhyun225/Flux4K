data_path="/data1/yhyun225/Diffusion4K/images"
save_path="/data1/yhyun225/Diffusion4K/extracted_images"

mkdir -p $save_path

for i in {0001..0105}; do
    echo "Extracting $i.tar"
    tar -xf "${data_path}/$i.tar" -C $save_path
done