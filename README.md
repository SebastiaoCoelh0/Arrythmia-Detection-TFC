# TFC 2024/2025 - Deteção de Ritmos Cardíacos Anormais em Registos de ECG

Trabalho Final de Curso (TFC) desenvolvido no âmbito da Licenciatura em Informática de Gestão na Universidade Lusófona,
sob orientação da Professora **Iolanda Velho**, coorientação do Professor **Lúcio Studer** e do
**Dr. Luís Rosário**.<ul />
Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de identificar **ritmos cardíacos
anormais** com base na **variabilidade da frequência cardíaca (HRV)**, a partir de **registos ECG de curta duração** (10
segundos).

## Dados Utilizados

- **Data Set**:
  [A Large Scale 12-lead Electrocardiogram Database for Arrhythmia Study](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
- **Formato**: Registos de ECG a 500Hz, com anotações de diagnóstico baseadas em códigos **SNOMED-CT**
- **Acesso**: Público e disponível para investigação científica

## Estrutura do Projeto

```bash
TFC/
├── data/
│   ├── WFDBRecords/     # Registos brutos descarregados da PhysioNet
│   ├── processed/       # Dados transformados (pickles, .csv, etc.)
├── df_ml/               # DataFrames finais prontos para treino/teste
├── models/              # Modelos treinados (.joblib)
│   └── afib_only/       # Modelos específicos para Fibrilação Auricular (AFIB)
├── notebooks/           # Jupyter Notebooks de exploração e análise
├── src/                 
│   ├── data/            # Carregamento de dados brutos
│   ├── features/        # Extração de métricas RR/HRV
│   ├── models/          # Treino e seleção de modelos
│   ├── pipelines/       # Scripts principais de execução
│   └── legacy/          # Scripts antigos não utilizados no pipeline atual
├── .gitignore
├── README.md
```

# TFC 2024/2025 - Detection of Abnormal Heart Rhythms in ECG Records

Final Project developed for the Bachelor's Degree in Information Technology Management at Universidade Lusófona, under
the
supervision of Professor Iolanda Velho, co-supervision of Professor Lúcio Studer and Dr. Luís Rosário<ul />
This project aims to develop a Machine Learning model capable of identifying
**abnormal heart rhythms** based on **Heart Rate Variability (HRV)**, using **short duration ECG records** (10 seconds).

## Dataset

- **Data Set**:
  [A Large Scale 12-lead Electrocardiogram Database for Arrhythmia Study](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
- **Format**: ECG recordings at 500Hz, with diagnostic annotations using **SNOMED-CT** codes
- **Access**: Public and available for scientific research

## Project Structure

```bash
TFC/
├── data/
│   ├── WFDBRecords/     # Raw records downloaded from PhysioNet
│   ├── processed/       # Transformed data (pickles, .csv, etc.)
├── df_ml/               # Final DataFrames for training/testing
├── models/              # Trained models (.joblib)
│   └── afib_only/       # Models specific to Atrial Fibrillation (AFIB)
├── notebooks/           # Jupyter Notebooks for exploration and analysis
├── src/                 
│   ├── data/            # Raw data loading
│   ├── features/        # RR/HRV metrics extraction
│   ├── models/          # Training and model selection
│   ├── pipelines/       # Main execution scripts
│   └── legacy/          # Old scripts not used in the current pipeline
├── .gitignore
├── README.md
```