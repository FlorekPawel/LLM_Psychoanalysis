# LLM_Psychoanalysis
Project repository for LLM psychoanalysis

# How to run
To run `main.py`, which is responsible for prompting the LLM, it is necessary to have the model running in the LM Studio application. Once the model is active, you can begin the prompting process by executing the following command in the console:

```bash
python main.py --results-dir model_name --loglevel log_level --end n_personas --batch_size batch_size

```
-	--results-dir model_name
The name of the model as registered in LM Studio. This also defines the directory where results will be stored.
-	--loglevel log_level
The logging level for terminal output. Values: DEBUG, INFO, WARNING, ERROR.
-	--end n_personas
The number of personas to use during prompting. The process will stop after generating prompts for this number of personas.
-	--batch_size batch_size
The size of each batch. Questionnaires will be split into batches of this size to optimize model performance.

plot_data.py will generate data necessary for plot generation from prompting results.
generate_plots.py will generate plots based on the results from the plot_data.py.
