# TFC 2024/2025 - Detecção de Fibrilhação Auricular (AFIB) em Registos de ECG

Este projeto tem como objetivo desenvolver um pipeline de Machine Learning capaz de detetar **Fibrilhação Auricular (AFIB)** a partir de **registos de ECG de curta duração** (10 segundos).

## Dados Utilizados
- Data Set: [A Large Scale 12-lead Electrocardiogram Database for Arrhythmia Study](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
- Registos de ECG em 500Hz, com anotações de diagnóstico baseadas em códigos SNOMED-CT.

## Estrutura do Projeto

```bash
TFC/
├── data/
│   ├── raw/          # Base de dados original
│   ├── processed/    # Dados processados e pickles
├── models/           # Modelos treinados (.joblib)
├── notebooks/        # Análises e exploração (Jupyter Notebooks)
├── src/              # Scripts Python principais
├── .gitignore
├── README.md
