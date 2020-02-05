Spiking Neural Gym agent created on top of NEURON++ and OpenAI Gym libraries

## Prerequisites

* Python 3.6

* Install requirements.txt

* Install NEURON from the instruction: https://github.com/ziemowit-s/neuron_netpyne_get_started

* Install OpenAI Gym Atari environments
    ```bash
    pip install gym[atari]
    ```

## MOD compilation
* Before run you must compile mod files and copy compiled folder to the main folder (where run Python files are located)
    ```bash
    nrmivmodl
    ```

* To help with compilation use compile_mod.py or CompileMOD class:
  * It will compile all mods inside the source folder (you can specify many source folders)
  * copy compiled folder to the target folder 
    ```bash
    python compile_mod.py --sources [SOURCE_FOLDER_WITH_MOD_FILES] --target [TARGET_FOLDER]
    ``` 
  * By default it works on Linux but you can change default params so that they express your OS params:
    * compiled_folder_name="x86_64"
    * mod_compile_command="nrnivmodl"
    
## Run

### pong_gym_run.py

* compile mods:
  * go to neuronpp_gym main folder and compile following mod folders in agents/utils/mods:
    * 4p_ach_da_syns
    * ebner2019
    * neuron_commons
    
    ```bash
    python compile_mod.py -source agents/utils/mods/4p_ach_da_syns agents/utils/mods/ebner2019 agents/utils/mods/neuron_commons -target .
    ``` 
    
### Known issues:

* After a long run it may terminate with:
```bash
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc

Process finished with exit code 134 (interrupted by signal 6: SIGABRT)
```

