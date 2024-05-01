# AdaFaceMin(imal)

## Prerequisites

- Python 3.x
- Poetry (optional)

## Installation

1. Clone the repository:

    ```shell
    git clone https://github.com/your-username/your-repo.git
    ```

2. Change into the project directory:

    ```shell
    cd your-repo
    ```

3. Install project dependencies using Poetry:

    ```shell
    poetry install
    ```

## Usage

1. Activate the virtual environment:

    ```shell
    poetry shell
    ```

2. Run the project:

    ```shell
    python main.py
    ```

## Creating a Virtual Environment with requirements.txt

1. Create a virtual environment:

    ```shell
    python -m venv venv
    ```

2. Activate the virtual environment:

    ```shell
    source venv/bin/activate
    ```

3. Install project dependencies from requirements.txt:

    ```shell
    pip install -r requirements.txt
    ```

4. Run the project:

    ```shell
    python train.py
    ```

    For heplp page see:
    
    ```shell
    python train.py --help
    ```
    
## Reference results 
### AdaFace iResNet101 (WebFace12M) Accuracy
- **lfw**      0.9981666666666665
- **cfp_fp**   0.9922857142857143
- **cplfw**    0.9458333333333334
- **calfw**    0.9601666666666668
- **agedb_30** 0.9776666666666666

### AdaFace iResNet101 (WebFace4M) Accuracy
- **lfw**      0.9981666666666668
- **cfp_fp**   0.9908571428571428
- **cplfw**    0.9446666666666668
- **calfw**    0.9605
- **agedb_30** 0.9786666666666667

### AdaFace iResNet101 (MS1M v2) Accuracy
- **lfw**      0.9981666666666665
- **cfp_fp**   0.9828571428571428
- **cplfw**    0.931
- **calfw**    0.9605
- **agedb_30** 0.9793333333333333 

### AdaFace iResNet101 (MS1M v3) Accuracy
- **lfw**      0.9976666666666667
- **cfp_fp**   0.9874285714285715
- **cplfw**    0.9366666666666665
- **calfw**    0.9611666666666666
- **agedb_30** 0.9814999999999999

