from util.consts import NUM_OF_HYPHENS
import inspect
import os
import time
import gdown


def download_drive_file(file_url, destination):
    gdown.download(file_url, destination, quiet=True)

def download_drive_models():
    func_name = inspect.currentframe().f_code.co_name
    print('-' * NUM_OF_HYPHENS)
    print(f'{func_name}...')
    for folder, files in FOLDER_TO_IDS.items():
        print(f'download: {folder}')
        dest_folder_path = f'./resources/models/{folder}'
        os.makedirs(dest_folder_path, exist_ok=True)
        for file_name, file_id in files.items():
            dest_file_path = os.path.join(dest_folder_path, file_name)
            src_file_url = f'https://drive.google.com/uc?id={file_id}'
            download_drive_file(src_file_url, dest_file_path)
            time.sleep(2)
    print(f'{func_name} finished!')




FOLDER_TO_IDS = {
    'painter_actor': {'actor.pkl': '10iUUNJMEhkPUh4muq48VXI1YS_kypTDG'},
    'painter_renderer': {'renderer.pkl': '1aakJYRpz_NT-bXdpgqNLSX4nVqqRBg2x'},
    'train_surrogate_painter': {'model_t50.pth': '17QU5s2K42MUVu9Ny9_BGW7nZvXG534DQ',
                                'model_t100.pth': '1xGwyy3ICmCm8rMyvC72nEVE_VYfD3y1u',
                                'model_t150.pth': '1ZLhCJA4668rs8H-jVRBd0y5vA_61B-kz',
                                'model_t200.pth': '1GrOgas4Wz_dkeImpI3Yvn-lYcTn6UJpr',
                                'model_t300.pth': '1gLxPu9E4WMwCop0mQwFvObStBVZuwKV3',
                                'model_t400.pth': '1oIbXDlhiHKRIVZGz5CSWlKzyUqrR4kcv',
                                'model_t500.pth': '1tSUs-qXSfJy_NPzf4I9zWyxMTNJE1hwc',
                                'model_t600.pth': '1YgzmSQfQie4k9ogSKEPrMor-WiX0GqZf',
                                'model_t700.pth': '1ohUX4q9Ot7PZ4BPdb_bc76AQ1qS2ZxIZ',
                                'model_t950.pth': '1F3JJBXkCpEjOKTUPCFbYH1JRAwGEbHEh',
                                'model_t1200.pth': '1DYm7Ye0u8u11HQjLwolueaEgPAi8-t1a',
                                'model_t1700.pth': '1rkQIA_ogJvkNBkTaa4IQrzgAmiQt8YT5',
                                'model_t2200.pth': '1LYVrNPE3LSCwaqjgIEaYzO2nLoZHZHgm',
                                'model_t2700.pth': '1PXjFmijfz1u5thC2Gc75zR_jI5sOIEf-',
                                'model_t3200.pth': '1KidSZJtO22oLz6cPUtLHsxYHCowJCbVq',
                                'model_t3700.pth': '1-YMr4WDwMv9apHTuWQ6uYd97VYgK52tm',
                                'model_t4200.pth': '1A39ZOs3Yg7XqsWlGyP3K60bp09_eXViH',
                                'model_t4700.pth': '1Dx3oWMaiWj8vvac9lvTGYZXeFZ6P6csU',
                                'model_t5200.pth': '11aK8jUBClVyVMjqdwRdXPCBhackvLIgX'},
    'train_decisioner_fc_fgsm': {'model.pth': '1Ht2rtSqm-yXzTQl8TViwv9UFdTM2Qo2c'},
    'train_decisioner_fc_fgsm_pgd': {'model.pth': '1BK3NaqyA5KygWXToNIG-m5zFRsoPPI_r'},
    'train_victim_clf_b': {'model.pth': '1z8Cnn00zNtKuCojArReX9--X8FBu0mz4'},
    'train_victim_clf_bp': {'model.pth': '1rb7yMOJQUbuIRkF-dETj-utjzPjXR5sT'}
}


download_drive_models()
