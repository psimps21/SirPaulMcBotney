emberrender --in="final_v1_anim.flam3" --opencl --verbose --progress --name_enable --sp
ffmpeg -framerate 25 -i "flames/"%03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "flames/"output.mp4
