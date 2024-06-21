import imageio
import tqdm


def create_gif(video_path, gif_path):
    video = imageio.get_reader(video_path)
    gif_writer = imageio.get_writer(gif_path, mode='I')
    for frame in video:
        gif_writer.append_data(frame)
    gif_writer.close()

video_path = 'I:/Chercheurs/Nguyen_DangKhoa/Projets_Recherche/Video/Video_original_mp4/p117-1.mp4'
gif_path = './Gif/EMU/output4.gif'

create_gif(video_path, gif_path)