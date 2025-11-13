# IA de Diagnóstico Hierárquico para Falhas Aeronáuticas

> Este projeto é um protótipo de um sistema de FID (Fault Detection and Identification) para aeronaves.

Em vez de usar um modelo de IA "monolítico", esta arquitetura usa um comitê de 13+ modelos especialistas (RandomForestClassifier) em uma cascata de 3 níveis. O objetivo é imitar o processo de diagnóstico de um engenheiro de manutenção: começar pelo sistema principal, ir para o subsistema e, finalmente, isolar o componente com falha.

`Nível 1 (Sistema)` → `Nível 2 (Subsistema)` → `Nível 3 (Diagnóstico Específico)`

Exemplo de Diagnostico: `SISTEMAS` → `HIDRÁULICO` → `VAZAMENTO BOMBA DE COMBUSTIVEL B`

## Motivos da utilização de uma arquitetura hierárquica

Um modelo "flat" (plano) que tenta adivinhar 1 de 50 falhas é ineficiente e ignora a relação óbvia entre os sistemas. A abordagem em cascata (conhecida como Local Classifiers Per Node) é mais *precisa*, mais *interpretável*, e *facilmente escalável*.

A arquitetura utiliza modelos especialistas em cascata, nos quais o resultado de um modelo serve de entrada para o próximo nível, garantindo uma análise refinada e interpretável. Cada nível possui modelos especializados, treinados individualmente com técnicas supervisionadas, ajustados para o seu domínio de atuação.

Pode-se dizer que neste projeto possuimos um protótipo de FDI (Fault Detection and Identification). Entretanto temos um conjunto total de 13+ modelos de IA (entre classificadores e filtros de decisão). 

## Estrutura dos Dados

Devido à óbvia confidencialidade dos dados reais de falhas, este projeto roda em um dataset sintético de alta fidelidade gerado pelo script `src/dataset_create.py`

Este gerador cria um "Digital Twin" do voo, produzindo um .csv com 71 features (sensores) e 3 labels (rótulos de diagnóstico).

- Features (X): `sensor_EGT_motor_1`, `sensor_pressao_hidr_A`, `sensor_volt_gerador_1`, etc.
- Labels(y): `nivel1`, `nivel2`, `nivel3`  
 
Devido à **restrição e confidencialidade de dados reais de falhas aeronáuticas**, as informações originais não são publicamente disponíveis. Por isso, foi criada uma **base simulada com 74 parâmetros de sensores** que reproduzem o comportamento de sistemas críticos durante o voo.

Esses parâmetros representam medidas típicas coletadas por sistemas embarcados, incluindo **variáveis ambientais, de navegação, motor, combustível, hidráulico, elétrico e estruturais**.  


**Exemplo de amostra:**

| sensor_temp_externa | sensor_gps_latitude | sensor_gps_longitude | sensor_gps_altitude | sensor_velocidade_gps | sensor_heading_magnetico | sensor_heading_gyro | nivel1 | nivel2 | nivel3 |
|---------------------:|--------------------:|----------------------:|--------------------:|----------------------:|-------------------------:|--------------------:|--------|--------|--------|
| -46.980416 | -23.501459 | -46.607012 | 35220.71 | 449.72 | 93.73 | 87.37 | NORMAL | NORMAL | NORMAL |
| -47.091172 | -23.501421 | -46.625197 | 35507.33 | 453.08 | 88.68 | 90.29 | ESTRUTURAL | FUSELAGEM | DANO POR FADIGA |
| -42.009555 | -23.501841 | -46.611753 | 34777.38 | 456.83 | 92.26 | 90.86 | SISTEMAS | MOTOR | EGT ALTA M2 |


## Organização do projeto

```bash
📦 diagnostico_hierarquico/
 ┣ 📁 data/
 │  ┣ 📁 processed/               
 │  ┗ 📁 raw/                      
 │     ┗ 📄 dataset_aircraft_failures.csv
 │
 ┣ 📁 models/
 │  ┣ 📁 nivel1/
 │  │  ┗ 📄 modelo_raiz.pkl
 │  ┣ 📁 nivel2/
 │  │  ┣ 📄 modelo_estrutural.pkl
 │  │  ┗ 📄 modelo_sistemas.pkl
 │  ┗ 📁 nivel3/
 │     ┣ 📄 modelo_asa.pkl
 │     ┣ 📄 modelo_avionica.pkl
 │     ┣ 📄 modelo_cabine.pkl
 │     ┣ 📄 modelo_combustivel.pkl
 │     ┣ 📄 modelo_controles_de_voo.pkl
 │     ┣ 📄 modelo_eletrico.pkl
 │     ┣ 📄 modelo_fuselagem.pkl
 │     ┣ 📄 modelo_hidraulico.pkl
 │     ┣ 📄 modelo_motor.pkl
 │     ┣ 📄 modelo_pneumatico.pkl
 │     ┗ 📄 modelo_trem_de_pouso.pkl
 │
 ┣ 📁 notebooks/
 │  ┗ 📄 exploratory_analysis.ipynb
 │
 ┣ 📁 src/
 │  ┣ 📄 dataset_create.py         # Criação e pré-processamento do dataset
 │  ┣ 📄 models.py                 # Definição e carregamento dos modelos
 │  ┗ 📄 predict.py                # Pipeline de previsão hierárquica
 │
 ┣ 📄 .gitignore
 ┣ 📄 requirements.txt
 ┗ 📄 README.md

```

## Como rodar a pipeline completa

1. Clone e instale:
    ```Bash
        git clone https://github.com/DanielMilanez/HAD.git
        cd HAD
        pip install -r requirements.txt
    ```

1. Gerar o dataset
    ```Bash
        python src/dataset_create.py
    ```

1. Treinar os modelos
    ```Bash
        python src/models.py
    ```

1. Fazer o diagnóstico
    ```
        python src/predict.py
    ```

