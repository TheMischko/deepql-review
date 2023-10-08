# DeepQ Learning Code review
### Úvod
Potřebuji od vás zkontrolovat zda se někde nenachazí chyba v mé implementaci agentů Deep Q-learningu.
První verze Deep Q-learningu byla představena v práci [Mnih et al.(2013)](dqn_2013.pdf).

Práce se zabývala propojením hlubokého učení (Hluboké neuronové sítě) a posilovaného učení (Q-learning).
Hluboké učení ale v rámci posilovaného učení přináší následující problémy:
- během života agenta se mění distribuce dat, ze které se učí, protože se učením mění jeho chování,
- sbírané zkušenosti přicházejí sekvenčně a jsou na sobě časově závislá.

Autoři tyto problémy vyřešili pomocí mechanismu Replay Memory, který slouží jako buffer, do kterého jsou ukládány zkušenosti agenta, a v učící fázi jsou z něj vytahávány náhodně uspořádané vzorky.
Data tak tvoří distribuci, která se v zásadě nemění (pořád se lze učit z nejstarších zkušeností) jen se rozrůstá, a zároveň na sobě nejsou jednotlivé vybrané prvky závislé.

Algoritmus tak zjednoduššeně pracuje takto:
1. Pozoruj nový stav a získej odměnu.
2. Ulož do Replay Memory starý stav, provedenou akci, nový stav a odměnu.
3. Vyber novou akci podle aktuálně naučené politiky (neuronové sítě).
4. Zapamatuj si nový stav a zvolenou akci do příští iterace.
5. Proveď učící algoritmus:
   1. Získej z Replay Memory náhodný vzorek zkušeností (batch).
   2. Podle aktuální politiky (neuronové sítě) vypočítej Q hodnotu pro páry stav-akce z učících vzorků.
   3. Pro stejné páry vypočítej Q hodnotu podle Bellmanovi rovnice.
   4. Výsledky z Bellmanovi rovnice použij jako Target hodnoty, vypočítej z nich gradient a proveď Zpětnou propagaci.
6. Proveď zvolenou akci a opakuj od bodu 1.

Neuronová síť v této architektuře zastává Q-funkci a učí se pro dvojice stav-akce jejich Q-hodnoty. Architektura této neuronové sítě má podobu:
- vstupní vrstva, která má velikost stavového vektoru (v původní práci je to snímek obrazovky Atari)
- (v původní práci kvůli práci s obrazovým vstupem obsahuje konvoluční vrstvy)
- skryté lineární vrstvy
- výstupní vrstva, která má velikost podle počtu možných akcí

V jednom dopředném průchodu jsme tak schopni zjistit Q hodnoty všech možných akcí pro aktuální stav.

Problém s touto architekturou je její nestabilita, protože v rámci učení se používá neuronová síť, jak pro odhad Q-hodnot, tak pro jejich cílovou hodnotu v rámci Bellmanovi rovnice.
Neuronová síť se tak trochu učí sama proti vlastním odhadům.
Rozšiřující práce [Mnih et al.(2015)](dqn_2015.pdf) proto přidává do architektury druhou neuronovou síť. Ta původní se nazývá *policy network* a nová síť *target network*.
Úlohou target network je v rámci učícího algoritmu generovat target hodnoty, ze kterých se policy network učí. Tyto dvě sítě se ale po uplynutí každých *n-kroků* synchonizují, tedy váhy z policy network se překopírují do target network.
Target network je tak obrazem historie policy network. Práce ukazuje, že tato architektura vede na stabilnější učení a lepší výsledky.

### Jaký je můj problém
V rámci své diplomové práce jsem vytvořil dva agenty. Jeden odpovídá architektuře z původního článku a obsahuje tak jednu neuronovou síť ([DQ_l.py](agents/DQ_l.py)).
Druhý agent odpovídá novější architektuře s dvěma neuronovými sítěmi ([DDQ_l.py](agents/DDQ_l.py)). 

Během experimentů se ukazuje, že oba agenti mají stejné skóre, respektive architektura s jednou neuronovou sítí má lehce vyšší výsledky. (V rámci intervalů spolehlivosti, se ale nedá říct, že je mezi agenty rozdíl.)
Očekávání podle původních prací jsou, že agent s dvěma neuronovými sítěmi by měl být signifikantně lepší než agent s jednou neuronovou sítí.

Chci si být tedy jistý, že až budu tvrdit, že agenti si vedou stejně, tak že nebude chyba v mém kódu. Důvod, proč si vedou stejně by zřejmě mohl být kvůli jednoduchosti problémů AIQ testu, na kterém se neukáže benefit dvou neuronových sítí.

### Co lze kde najít
- Jelikož mezi oběma agenty byl značný překryv společné logiky, tak jsem ji vyextrahoval do třídy, ze které oba agenti dědí - [agents/deepq_l/IDeepQLAgent.py](agents/deepq_l/IDeepQLAgent.py)
  - Neuronová síť je vytažena bokem do [agents/deepq_l/neural_net.py](agents/deepq_l/neural_net.py)
  - Replay Memory je v  [agents/deepq_l/replay_memory.py](agents/deepq_l/replay_memory.py)
  - Oba agenti ještě využívají nadstavbu Q-learningu v podobě Eligiblity traces  [traces.py](agents/deepq_l/traces.py)
- Agent s jednou neuronovou sítí - [agents/DQ_l.py](agents/DQ_l.py)
- Agent s dvěmi neuronovými sítěmi - [agents/DDQ_l.py](agents/DDQ_l.py)

Já vás žádám hlavně o kontrolu souborů [IDeepQLearning.py](agents/deepq_l/IDeepQLAgent.py), [DQ_l.py](agents/DQ_l.py), [DDQ_l.py](agents/DDQ_l.py), kde se nachází hlavní logika agentů.

### Co dalšího vědět o kódu
- Stavový vektor není brán jako stav získaný od prostředí, ale jako kombinace aktuálního stavu a stavu těsně předchozího. Stavový vektor tak obsahuje nejdříve prvky předchozího stavu a poté prvky aktuálního stavu. Tento stavový vektor tvoří funkce *transfer_observation_to_state_vec* v rámci [IDeepQLAgent.py](agents/deepq_l/IDeepQLAgent.py)
- Jak často agent vybírá náhodné akce (a prozkoumává tak prostředí) ovlivňuje parametr *epsilon*. Agenti pracují ve dvou módech. První mód bere epsilon jako statickou hodnotu a v průběhu života agenta se nemění. V druhém módu začíná *epsilon* na hodnotě 1.0 a po dobu nastaveného počtu iterací se hodnota *epsilon* lineárně snižuje. Proto v kódu existují funkce jako *decrement_epsilon*.
- V kódu také jsou pozůstatky mé snahy vykreslovat grafy učení, pro účely debuggování. Vše co obsahuje slovo *"plot"* tak můžeme vypustit.

### Jak agenty spustit
Následují dva příkazy, které agenty spustí s hodnotami parametrů z původních prací.

Pro DQ_l:
```shell
python AIQ.py -r BF -a DQ_l,0.00025,0.99,32,1,2000,64,512,0,1,0,0 -l 1000 -s 100
```

Pro DDQ_l:
```shell
python AIQ.py -r BF -a DDQ_l,0.00025,0.99,32,1,2000,64,512,0,1,1,200,0,0 -l 1000 -s 100
```