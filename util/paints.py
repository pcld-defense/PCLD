import subprocess
from LearningToPaint.baseline.test_new_from_drive import main, Arguments


def paint(max_step, img_path, divide, output_dir, output_canvas_dir, output_img_name, save_every, save_canvas_every,
          save_strokes=False, strokes_dir=None, verbose=False):
    run_command = ['python3', 'LearningToPaint/baseline/test_new_from_drive.py',
                   '--max_step', str(max_step),
                   '--actor', 'LearningToPaint/actor.pkl',
                   '--renderer', 'LearningToPaint/renderer.pkl',
                   '--img', img_path,
                   '--divide', str(divide),
                   '--output_dir', output_dir,
                   '--output_canvas_dir', output_canvas_dir,
                   '--output_img_name', output_img_name,
                   '--save_every', str(save_every),
                   '--save_canvas_every', str(save_canvas_every)
                   ]
    if save_strokes:
        run_command.append('--save_strokes')
        run_command.append('true')
        run_command.append('--strokes_dir')
        run_command.append(strokes_dir)
    if verbose:
        run_command.append('--verbose')
        run_command.append('true')

    subprocess.run(run_command)


def paint_image(max_step, img_path, divide, output_dir, output_canvas_dir, output_img_name, save_every, save_canvas_every,
                save_strokes=False, strokes_dir=None, verbose=False):
    args = Arguments(
        max_step=max_step,
        actor='LearningToPaint/actor.pkl',
        renderer='LearningToPaint/renderer.pkl',
        img=img_path,
        imgid=0,
        divide=divide,
        output_dir=output_dir,
        output_canvas_dir=output_canvas_dir,
        output_img_name=output_img_name,
        save_every=save_every,
        save_canvas_every=save_canvas_every,
        save_strokes=save_strokes,
        strokes_dir=strokes_dir,
        verbose=verbose
    )

    main(args=args)
