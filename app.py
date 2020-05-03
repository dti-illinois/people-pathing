import sys

from people_pathing import PeoplePathing


def show_usage():
    print('usage: python app.py video_file_path [overlay_image_path] [-n=num_frames] [-s, --silent] [-d, --detection]')


def start_people_pathing(args):
    if len(args) < 2 or len(args) >= 7:
        show_usage()
        return
    overlay_image_path = None
    num_frames = None
    silent = False
    show_detection = False
    for arg in args[2:]:
        if arg[0:3] == '-n=':
            num_frames = int(arg[3:])
        elif arg == '-s' or arg == '--silent':
            silent = True
        elif arg == '-d' or arg == '--detection':
            show_detection = True
        else:
            overlay_image_path = arg

    pp = PeoplePathing()
    image_size, paths = pp.get_paths(args[1],
                                     num_frames=num_frames,
                                     show_detection=show_detection,
                                     silent=silent)
    if overlay_image_path:
        pp.plot_paths_on_image(image_size, paths, overlay_image_path)
    else:
        pp.plot_paths(image_size, paths)


if __name__ == '__main__':
    start_people_pathing(sys.argv)
