
## Requirements
- Python >= 3.6
- Pytorch >= 1.7

## Folder Specification

- **ddi_main.py:** Please excuate this file to run our DDI model.
- **ddi_predictor5.py:** It includes our main architecture code.
- **Dataset** This folder includes **deep_drug_dict.json** containing all dictionary lists of drugs, **deep_train.json, deep_valid.json, and deep_test.json,** including annotation and unannotated relationship information between pair drugs.
- **Utils:**  This folder includes **ddi_get_mol_graphs.py** responsible for transforming drug SMOLES sequence data into drug moleculers and **ddi_dataset_wo_type.py** splitting datasets into train, validated and test datasets for DDI.
- **Utils:**  This folder includes **ddi_layers.py** containing import representation learning blocks and **ddi_train.py** optimizing framework via chi-square.
## Run the Code
  To excecute this framework, please run the following command. The contents include train and inference:

```bash
cd cd
python ddi_main.py
``` 
## Acknowledgement
We sincerely thank Weiyu Shi for providing code. Please contact us with Email: standyshi@qq.com
<u><p><b><i><font size="6">If you are interested in Natural Language Processing and Large Language Models, feel free to contact us by Email: zhangyijia@dlmu.edu.cn </font></i></b></p>


