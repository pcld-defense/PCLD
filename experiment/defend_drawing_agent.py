import concurrent.futures
from util.consts import NUM_OF_HYPHENS
from util.paints import *
from util.models import *
import os


def paint_dataset(old_local_root, save_every, actor, renderer):
    new_local_root = old_local_root + '_paints'
    futures = []
    # with concurrent.futures.ThreadPoolExecutor(5) as executor:
    with concurrent.futures.ThreadPoolExecutor(20) as executor:
        i = 0
        for subdir, dirs, files in os.walk(old_local_root):
            if files:
                subdir_new = os.path.join(new_local_root, '/'.join(subdir.split('/')[2:]))
                for file in files:
                    if '.DS_Store' in file:
                        continue
                    im_name = file.split('/')[-1].split('.')[0]
                    old_file_path = os.path.join(subdir, file)
                    os.makedirs(os.path.dirname(subdir_new), exist_ok=True)
                    future = executor.submit(paint_image,
                                             max_step=80,
                                             actor=actor,
                                             renderer=renderer,
                                             img_path=old_file_path,
                                             divide=5,
                                             output_dir=subdir_new,
                                             output_canvas_dir='',
                                             output_img_name=im_name,
                                             save_every=save_every,
                                             save_canvas_every='',
                                             save_strokes=False,
                                             strokes_dir=None,
                                             verbose=False)
                    futures.append(future)
                    i += 1
                    if i % 10 == 0:
                        print(i)
        # Wait for all the functions to complete
        print("start paint")
        for future in concurrent.futures.as_completed(futures):
            # Get the result of the function call
            result = future.result()
            i += 1
            if i % 10 == 0:
                print(i)
    return new_local_root


def paint_attacked_dataset_iterative(old_local_root, new_local_root, save_every):
    i = 0
    for subdir, dirs, files in os.walk(old_local_root):
        if files:
            subdir_new = subdir.replace(old_local_root, new_local_root)
            for file in files:
                if '.DS_Store' in file:
                    continue
                im_name = file.split('/')[-1].split('.')[0]
                old_file_path = os.path.join(subdir, file)
                os.makedirs(os.path.dirname(subdir_new), exist_ok=True)
                # generate paints and save them
                paint_image(max_step=80,
                            img_path=old_file_path,
                            divide=5,
                            output_dir=subdir_new,
                            output_canvas_dir='',
                            output_img_name=im_name,
                            save_every=save_every,
                            save_canvas_every='',
                            save_strokes=False,
                            strokes_dir=None,
                            verbose=False)
                # save the original input as well
                Image.open(old_file_path).save(subdir_new+'/'+im_name+'.png')
                i += 1
                if i % 100 == 0:
                    print(f'finished painting {i+1} images')
    return new_local_root


def main_defend_drawing_agent(ds_local_path: str, save_every: str):
    print('-' * NUM_OF_HYPHENS)
    print('Running defend using drawing agent experiment...')

    if not os.path.exists(ds_local_path):
        raise Exception(f'Folder not exist: {ds_local_path}')

    print('start drawing')
    new_local_root = ds_local_path+'_paints'
    new_local_root = paint_attacked_dataset_iterative(ds_local_path, new_local_root, save_every)
    print('drawing completed!')

    print('Finished defend using drawing agent experiment!')
