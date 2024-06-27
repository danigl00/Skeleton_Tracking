import imageio

def gif_to_mp4(gif_path, mp4_path):
    try:
        with imageio.get_reader(gif_path) as reader:
            fps = reader.get_meta_data()['fps']
            writer = imageio.get_writer(mp4_path, fps=fps)
            for frame in reader:
                writer.append_data(frame)
            writer.close()
        print("Conversion successful!")
    except Exception as e:
        print(f"Conversion failed: {str(e)}")

inpit = "C:/Users/p0121182/Project/Skeleton_Tracking/Sample_gifs/Spiderman3.gif"
output = "sm.mp4"
# Usage example
gif_to_mp4("input.gif", "output.mp4")