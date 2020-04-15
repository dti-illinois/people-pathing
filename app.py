# pylint: disable-all

from people_pathing import PeoplePathing

if __name__ == '__main__':
    pp = PeoplePathing()
    paths = pp.get_paths('data/videos/people_walking_mall.mp4',
                         show_detection=False)
    pp.plot_object_paths(paths)
    pp.plot_object_paths_on_image(paths, 'data/images/frame_1.jpg')
