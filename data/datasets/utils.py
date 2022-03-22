import os
from collections import defaultdict
from data.datasets.meta_classes import DataInfo, ValInfo, DataInfoSemantic, MetaInfoSemantic
import glob

IMAGE_EXTENSION = (".jpg", ".jpeg", ".png")

def get_raw_datasets(cfg):
    root_dir = cfg.training.datasets.root_dir
    root_folder = SemanticFolder()

    datasets = defaultdict(list)

    sub_folder_lists = [x for x in os.listdir(root_dir)]

    for sub_folder_name in sub_folder_lists:
        sub_folder = root_folder.get_sub_folder(sub_folder_name)
        sub_folder.add_meta(os.path.join(root_dir, sub_folder_name))

    # dataset_type: 각 train / val 의 하위 폴더에 저장된 모든 폴더의 정보를 병합
    for dataset_type, sub_folder in root_folder.sub_folder.items():
        # meta_info: image_dicts / mask_dicts / names

        for image_name in sub_folder.meta_folder.names:
            datasets[dataset_type].append(
                sub_folder.add_path(
                    image_path=sub_folder.meta_folder.image_dicts[image_name],
                    mask_path=sub_folder.meta_folder.mask_dicts[image_name]
                )
            )

    return datasets



def get_file_dicts(path: str ,ext: str = None)-> dict:
    info = defaultdict(str)
    image_extension = ext if ext is not None else IMAGE_EXTENSION
    for x in os.listdir(path):
        if x.endswith(image_extension):
            name, ext = os.path.splitext(x)
            info[name] = os.path.join(path, x)
    return info

def get_image_lists(path: str, ext: str = None) -> list:
    image_extension = ext if ext is not None else IMAGE_EXTENSION
    return [os.path.join(path, x) for x in os.listdir(path) if x.endswith(IMAGE_EXTENSION)]

class SemanticSubFolder:
    def __init__(self):
        pass

    def add_path(self, image_path: str, mask_path: str):
        return MetaInfoSemantic(image_path=image_path, mask_path=mask_path)

    def add_meta(self, path: str):
        """
            :param path: root_dir + train or val folder path 전달
                이 위치의 folder 에는 image 와 mask 이미지가 저장되어 있음
                - image 확장자는 jpg
                - mask 확장자는 png
            1. get file lists
                a. image 폴더내에 있는 파일명 리스트 얻어오기
                    예시) a : /workspace/datasets/transys/image/a.jpg
                b. mask 폴더내에 있는 파일명 리스트 얻어오기
                    예시) a : /workspace/datasets/transys/mask/a.png
            2. add #1 info into dataclass

        :return:
        """

        image_dicts = get_file_dicts(os.path.join(path, 'image'))
        mask_dicts = get_file_dicts(os.path.join(path, 'mask'))
        names = [name for name in image_dicts.keys()]

        self.meta_folder = DataInfoSemantic(image_dicts = image_dicts, mask_dicts=mask_dicts, names=names)

class SemanticFolder:
    """
            configs 에 있는 ROOT_DIR 의 폴더 정보를 저장하는 클래스
    """
    def __init__(self):
        self.sub_folder = defaultdict(SemanticSubFolder)

    def get_sub_folder(self, name: str):
        """
            :param name: sub folder 명
        :return: sub folder에 있는 이미지경로와 이미지갯수가 저장된 데이터클래스 정보 반환
        """
        return self.sub_folder[name]



class SubFolder:
    """
        configs 에 있는 ROOT_DIR 하위에 있는 폴더정보를 저장하는 클래스
        TODO: 이 부분은 차후 데이터 폴더구조를 어떻게 가져갈건지에 따라 변경
    """
    def __init__(self):
        self.meta_folder = defaultdict(DataInfo)

    def add_meta(self, path: str):
        """
            :param path: ROOT_DIR 경로
        ROOT_DIR 아래 저장된 sub folder 의 이미지를 스크랩하는 함수
        """
        target_dirs = [target for target in os.listdir(path)]

        for target in target_dirs:
            target_path = os.path.join(path, target)
            image_paths = get_image_lists(target_path)
            image_nums = len(image_paths)

            self.meta_folder[target] = DataInfo(image_paths, image_nums)

class Folder:
    """
        configs 에 있는 ROOT_DIR 의 폴더 정보를 저장하는 클래스
    """
    def __init__(self):
        self.sub_folder = defaultdict(SubFolder)

    def get_sub_folder(self, name: str):
        """
            :param name: sub folder 명
        :return: sub folder에 있는 이미지경로와 이미지갯수가 저장된 데이터클래스 정보 반환
        """
        return self.sub_folder[name]


class ValInferResults:
    """
        validation 에 대한 정보를 저장하는 클래스
    """
    def __init__(self):
        self.results = defaultdict(ValInfo)

    def add_info(self, name: str, loss: dict, model_path: str, model):
        """
            :param name: 실험순서(epoch)
            :param loss: loss 값
            :param model_path: 모델 저장경로
            :param model: 모델 state_dict
        """
        # dice, precision, recall, f_score, loss
        self.results[name] = ValInfo(loss= loss, model_path= model_path, model = model)


    def sort_select_metrics(self, selector: str, reverse=False):
        self.sorted_results = sorted(self.results.items(), key=lambda x: x[1].loss.get(selector), reverse=reverse)

    def sort_min_loss(self):
        """
            validation 실험결과 중 loss 가 제일 낮은 순서대로 정렬
        """
        self.sorted_results = sorted(self.results.items(), key=lambda x: x[1].loss , reverse=False)

    def get_final_model(self):
        """
            :return: validation loss 가 제일 낮은 실험결과에 대한 값을 반환
        """
        return {
            "name": self.sorted_results[0][0],
            "loss": self.sorted_results[0][1].loss,
            "model_path": self.sorted_results[0][1].model_path,
            "model": self.sorted_results[0][1].model
        }

def make_file_lists(root_dir: str) -> dict:
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f'There is no folder about {root_dir}')

    file_lists = defaultdict(str)

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext in IMAGE_EXTENSION:
                file_lists[name] = os.path.join(root, file)

    return file_lists