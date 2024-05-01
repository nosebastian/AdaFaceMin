from pathlib import Path
import numpy as np
import shutil

data_root: Path = Path('data') / 'raw'
train_test_val_ratio: tuple[float, float] = (0.8, 0.2)

train_test_val_ratio_np = np.array((0.0,) + train_test_val_ratio)
train_test_val_ratio_np /= train_test_val_ratio_np.sum()
train_test_val_ratio_np = train_test_val_ratio_np.cumsum()
all_data: list[Path] = list(data_root.glob('**/*.jpg'))
all_parent: set[Path] = set(data.parent for data in all_data)
random_state = np.random.RandomState(42)
random_values = random_state.rand(len(all_data))
random_assignment = np.digitize(random_values, train_test_val_ratio_np)
train_data: list[Path] = [data for data, assignment in zip(all_data, random_assignment) if assignment == 1]
val_data: list[Path] = [data for data, assignment in zip(all_data, random_assignment) if assignment == 2]
test_data: list[Path] = [data for data, assignment in zip(all_data, random_assignment) if assignment == 3]
print(f"{len(train_data) = }" )
print(f"{len(val_data) = }" )
print(f"{len(test_data) = }" )

for source, target in zip([train_data, val_data, test_data], ['train', 'val']):
    target_root: Path = data_root.parent / target
    for parent in all_parent:
        target_parent: Path = target_root / parent.relative_to(data_root)
        target_parent.mkdir(parents=True, exist_ok=True)
    for data in source:
        target_path: Path = target_root / data.relative_to(data_root)
        shutil.copy(data, target_path)
