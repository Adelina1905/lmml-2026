"""Synthetic report generator with broader vocabulary (my own templates, no real records).
Template syntax: {a|b|c} picks one option; <N> a number; <PLACE>, <RIVER> slots."""
import random, re

PLACES_EN = ["the town", "the city", "the village", "the region", "the valley", "the county", "the township", "the district",
             "the community", "the reserve", "the coast", "the island", "the province", "the north shore", "the interior"]
PLACES_FR = ["la ville", "le village", "la région", "la vallée", "le comté", "la municipalité", "la localité",
             "la communauté", "la réserve", "la côte", "l'île", "la province", "la rive nord", "l'arrière-pays"]
RIVERS = ["Red", "Fraser", "Saint John", "Assiniboine", "Souris", "Bow", "Elbow", "Thames", "Richelieu", "Chaudière",
          "Saguenay", "Petitcodiac", "Grand", "Qu'Appelle", "Peace", "Athabasca", "Ottawa", "Saint-François", "Yamaska", "Kennebecasis"]

H = {
"flood": {
"en": [
"{Spring runoff|Rapid snowmelt|Heavy rain|Torrential rain|Days of rain} caused the {<RIVER> River|river|creek|lake} to {overflow its banks|flood|rise rapidly|spill over|break through the dikes}.",
"{Flooding|Severe flooding|Widespread flooding|Flash flooding|The worst flood in decades} {hit|struck|affected|swamped} {<PLACE>|low-lying areas|farmland|the downtown core}.",
"An ice jam on the <RIVER> River {caused|produced|led to} flooding {in <PLACE>|upstream|in low-lying areas}.",
"{The river|The <RIVER> River|Water levels} crested {at a record level|well above flood stage|several metres above normal}.",
"{Hundreds|Thousands|Dozens} of {homes|basements|farms|cottages} were {flooded|under water|inundated|submerged}.",
"{Floodwaters|High water|Rising water} {washed out|damaged|closed|swept away} {roads and bridges|the highway|a bridge|the railway line}.",
"{Volunteers|Soldiers|Residents} {filled sandbags|built dikes|reinforced the dikes|sandbagged homes} to hold back the {water|floodwaters|river}.",
"A dam {burst|failed|was overtopped} after heavy rains, sending a wall of water {downstream|into the valley}.",
"Storm sewers {backed up|overflowed} and {thousands of basements|many streets} {were flooded|filled with water}.",
"Farmland was under water for {weeks|days|months} and seeding was delayed.",
"A storm surge and high tides flooded {the waterfront|coastal roads|low-lying neighbourhoods}.",
"The flood {forced|led to} the evacuation of {<N>|hundreds of|thousands of} residents.",
],
"fr": [
"{La crue printanière|La fonte rapide des neiges|Des pluies abondantes|Des pluies diluviennes|Plusieurs jours de pluie} {font déborder|gonflent} {la rivière <RIVER>|la rivière|le ruisseau|le lac}.",
"{Des inondations|De graves inondations|Des inondations généralisées|Des inondations soudaines|La pire inondation depuis des décennies} {frappent|touchent|submergent} {<PLACE>|les basses terres|les terres agricoles|le centre-ville}.",
"Un embâcle sur la rivière <RIVER> {provoque|cause} des inondations {dans <PLACE>|en amont|dans les secteurs bas}.",
"{Le niveau de la rivière|La rivière <RIVER>|Le niveau de l'eau} atteint {un niveau record|son sommet|plusieurs mètres au-dessus de la normale}.",
"{Des centaines|Des milliers|Des dizaines} de {maisons|sous-sols|fermes|chalets} sont {inondés|sous l'eau|submergés|envahis par les eaux}.",
"{Les eaux|Les hautes eaux|La crue} {emportent|endommagent|coupent} {des routes et des ponts|la route|un pont|la voie ferrée}.",
"{Des bénévoles|Des soldats|Des résidents} {remplissent des sacs de sable|érigent des digues|renforcent les digues} pour contenir {les eaux|la rivière|la crue}.",
"Un barrage {cède|se rompt|déborde} après de fortes pluies et libère un torrent d'eau {en aval|dans la vallée}.",
"Les égouts {refoulent|débordent} et {des milliers de sous-sols|de nombreuses rues} sont inondés.",
"Les terres agricoles restent sous l'eau pendant des {semaines|jours|mois}.",
"Une onde de tempête et de grandes marées inondent {le front de mer|les routes côtières|les quartiers bas}.",
"L'inondation force l'évacuation de {<N>|centaines de|milliers de} résidents.",
]},
"storm": {
"en": [
"A {tornado|powerful tornado|F2 tornado|F3 tornado|series of tornadoes} {touched down|tore through|struck|ripped through} {<PLACE>|the outskirts of <PLACE>|farmland}, {destroying|damaging|levelling} {homes|dozens of houses|barns|a church}.",
"A {severe thunderstorm|violent thunderstorm|hailstorm|summer storm} {brought|produced|dumped} {hail the size of golf balls|large hail|torrential rain and hail|hailstones} on {<PLACE>|the city|crops}.",
"{Hail|Large hail|Hailstones} {damaged|destroyed|flattened} {cars and roofs|crops|thousands of vehicles|greenhouses}.",
"{Hurricane|Tropical storm|Post-tropical storm|The remnants of hurricane} {Edna|Hazel|Juan|Igor|Fiona|Dorian|Arthur|Hilda} {struck|hit|swept across|made landfall on} {the coast|<PLACE>|the province} with winds of <N> km/h.",
"{Winds|Wind gusts|Gale-force winds|Hurricane-force winds} of {up to |over |}<N> km/h {uprooted trees|downed power lines|tore off roofs|knocked out power}.",
"A {windstorm|wind storm|violent storm|powerful storm} {left|cut power to} <N> {customers|homes} without electricity.",
"A storm surge {flooded|battered|damaged} {wharves|the waterfront|coastal homes|fishing boats}.",
"Lightning {struck|hit|started fires in} {several buildings|a barn|homes} during the storm.",
"A downburst {flattened|damaged|destroyed} {barns|grain bins|trees|mobile homes}.",
"The storm {destroyed|damaged|wrecked} {fishing boats|wharves|homes|power lines} along {the coast|the shore}.",
"{Heavy rain|Torrential rain} from the storm caused local flooding.",
"The tornado's path was {<N> km long|several hundred metres wide}.",
],
"fr": [
"Une {tornade|puissante tornade|tornade de catégorie F2|tornade de catégorie F3|série de tornades} {touche le sol|traverse|frappe|ravage} {<PLACE>|les abords de <PLACE>|les terres agricoles} et {détruit|endommage|rase} {des maisons|des dizaines de maisons|des granges|une église}.",
"Un {violent orage|orage violent|orage de grêle|orage d'été} {apporte|laisse tomber|s'abat avec} {des grêlons de la taille de balles de golf|de la grêle|des pluies torrentielles et de la grêle} sur {<PLACE>|la ville|les cultures}.",
"{La grêle|De gros grêlons|Les grêlons} {endommagent|détruisent|couchent} {les voitures et les toitures|les récoltes|des milliers de véhicules|des serres}.",
"{L'ouragan|La tempête tropicale|La tempête post-tropicale|Les restes de l'ouragan} {Edna|Hazel|Juan|Igor|Fiona|Dorian|Arthur|Hilda} {frappe|balaie|touche} {la côte|<PLACE>|la province} avec des vents de <N> km/h.",
"Des {vents|rafales|vents violents|vents de force ouragan} de {plus de |jusqu'à |}<N> km/h {déracinent des arbres|abattent des fils électriques|arrachent des toits|coupent le courant}.",
"Une {tempête de vent|violente tempête|forte tempête} prive <N> {abonnés|foyers} d'électricité.",
"Une onde de tempête {inonde|frappe|endommage} {les quais|le front de mer|les maisons côtières|les bateaux de pêche}.",
"La foudre {frappe|atteint|allume des incendies dans} {plusieurs bâtiments|une grange|des maisons} pendant l'orage.",
"Une rafale descendante {couche|endommage|détruit} {des granges|des silos|des arbres|des maisons mobiles}.",
"La tempête {détruit|endommage} {des bateaux de pêche|des quais|des maisons|des lignes électriques} le long de la côte.",
"Les pluies torrentielles de la tempête causent des inondations locales.",
"La trajectoire de la tornade s'étend sur {<N> km|plusieurs centaines de mètres}.",
]},
"winter": {
"en": [
"{Freezing rain|An ice storm|Days of freezing rain} {coated|covered|encased} {power lines and trees|roads and wires|the region} {with ice|in thick ice|with a heavy layer of ice}.",
"{Transmission towers|Hydro poles|Power lines} {collapsed|snapped|came down} under the weight of the ice.",
"A {blizzard|severe blizzard|three-day blizzard|winter storm|snowstorm} with {winds of <N> km/h|blowing snow|whiteout conditions} {paralysed|shut down|buried|hit} {<PLACE>|the city|the region}.",
"{A record|Heavy|More than <N> cm of} snowfall {buried|blanketed|fell on} {<PLACE>|the city|the region}.",
"{Roofs|Several roofs|A hockey arena roof|Barns} collapsed under the weight of the snow.",
"{An extreme cold snap|A cold wave|Bitter cold|Extreme cold} with temperatures {near|below|of} -<N> C {caused|led to} {frozen pipes|deaths from exposure|frostbite|power failures}.",
"{Motorists|Hundreds of motorists|Travellers} were stranded {in snowdrifts|on the highway|for days} {by the storm|in the blizzard}.",
"{Snow and ice|Blowing snow|Icy roads} {closed|shut down} {the highway|airports and highways|schools and businesses}.",
"{Millions|Hundreds of thousands|Thousands} were left without power {in the cold|for days|for weeks} after the ice storm.",
"The army was called in to {clear snow|help remove snow|help restore power}.",
"Sleet and freezing drizzle made roads treacherous.",
"Snowdrifts several metres high blocked roads and railways.",
],
"fr": [
"{La pluie verglaçante|Une tempête de verglas|Plusieurs jours de pluie verglaçante} {recouvre|enveloppe} {les lignes électriques et les arbres|les routes et les fils|la région} d'une {épaisse |}couche de glace.",
"{Des pylônes|Des poteaux électriques|Des lignes électriques} {s'effondrent|cèdent|se brisent} sous le poids de la glace.",
"Un {blizzard|violent blizzard|blizzard de trois jours|tempête hivernale|tempête de neige} avec {des vents de <N> km/h|de la poudrerie|une visibilité nulle} {paralyse|immobilise|ensevelit|frappe} {<PLACE>|la ville|la région}.",
"{Des chutes de neige record|D'abondantes chutes de neige|Plus de <N> cm de neige} {ensevelissent|recouvrent|tombent sur} {<PLACE>|la ville|la région}.",
"{Des toits|Plusieurs toits|Le toit d'un aréna|Des granges} s'effondrent sous le poids de la neige.",
"{Une vague de froid extrême|Un froid intense|Un froid polaire|Un froid extrême} avec des températures {de|sous les|près de} -<N> C {cause|entraîne} {le gel des conduites|des décès dus au froid|des engelures|des pannes de courant}.",
"Des {automobilistes|centaines d'automobilistes|voyageurs} restent bloqués {dans les congères|sur l'autoroute|pendant des jours} {par la tempête|dans le blizzard}.",
"{La neige et la glace|La poudrerie|Les routes glacées} {forcent la fermeture de|ferment} {l'autoroute|les aéroports et les autoroutes|les écoles et les commerces}.",
"{Des millions|Des centaines de milliers|Des milliers} de personnes sont privées d'électricité {dans le froid|pendant des jours|pendant des semaines} après la tempête de verglas.",
"L'armée est appelée pour {déneiger les rues|aider au déneigement|aider à rétablir le courant}.",
"Le grésil et la bruine verglaçante rendent les routes dangereuses.",
"Des bancs de neige de plusieurs mètres bloquent les routes et les voies ferrées.",
]},
"wildfire": {
"en": [
"A {forest fire|wildfire|bush fire|grass fire|brush fire} {burned|consumed|destroyed} {<N> hectares|thousands of hectares|large areas} of {forest|timber|bush|grassland} {near <PLACE>|in the area|north of <PLACE>}.",
"{Lightning|A campfire|Sparks from a train|Hot, dry weather} {started|ignited|sparked} {a forest fire|dozens of wildfires|a wildfire|a grass fire}.",
"The {fire|wildfire|blaze} {forced the evacuation of|threatened|reached the outskirts of|swept into} {<PLACE>|the town|several communities}.",
"{Water bombers|Hundreds of firefighters|Helicopters and water bombers|Fire crews} {fought the blaze|battled the fire|tried to contain the wildfire}.",
"{Smoke|Thick smoke|Smoke from the fires} {blanketed|covered|drifted over} {the region|<PLACE>|the province} for {days|weeks}.",
"Residents were {airlifted out|evacuated by air|forced to flee} as the {forest fire|wildfire} approached.",
"The {wildfire|fire} destroyed {<N> homes|a sawmill|cabins and cottages|the town} before it was brought under control.",
"{Strong winds|Gusty winds|Dry winds} drove the {fire|flames|wildfire} {through the forest|towards the town|across the hills}.",
"{The fire season|A severe fire season|The summer} saw {<N> fires|hundreds of fires} burn {<N> hectares|millions of hectares} in the province.",
"The fire was finally contained {after rain fell|by fire crews|weeks later}.",
"{Firebreaks|Fire guards} were bulldozed to protect the community.",
"The {wildfire|forest fire} jumped the highway and burned through the subdivision.",
],
"fr": [
"Un {incendie de forêt|feu de forêt|feu de broussailles|feu d'herbe|feu de brousse} {brûle|ravage|détruit} {<N> hectares|des milliers d'hectares|de vastes étendues} de {forêt|bois|broussailles|prairie} {près de <PLACE>|dans le secteur|au nord de <PLACE>}.",
"{La foudre|Un feu de camp|Des étincelles d'un train|Le temps chaud et sec} {allume|déclenche|provoque} {un incendie de forêt|des dizaines de feux de forêt|un feu de forêt|un feu d'herbe}.",
"L'{incendie|incendie de forêt|feu} {force l'évacuation de|menace|atteint la périphérie de|envahit} {<PLACE>|la ville|plusieurs collectivités}.",
"{Des avions-citernes|Des centaines de pompiers|Des hélicoptères et des bombardiers d'eau|Des équipes de lutte} {combattent les flammes|luttent contre l'incendie|tentent de maîtriser le feu}.",
"{La fumée|Une épaisse fumée|La fumée des incendies} {recouvre|enveloppe|envahit} {la région|<PLACE>|la province} pendant {des jours|des semaines}.",
"Les résidents sont {évacués par avion|évacués par voie aérienne|forcés de fuir} à l'approche de l'incendie de forêt.",
"L'incendie détruit {<N> maisons|une scierie|des chalets|la ville} avant d'être maîtrisé.",
"{Des vents forts|Des rafales|Des vents secs} poussent {le feu|les flammes|l'incendie} {à travers la forêt|vers la ville|sur les collines}.",
"{La saison des feux|Une saison des feux difficile|L'été} voit {<N> incendies|des centaines de feux} brûler {<N> hectares|des millions d'hectares} dans la province.",
"Le feu est finalement circonscrit {après la pluie|par les pompiers forestiers|des semaines plus tard}.",
"Des coupe-feu sont creusés pour protéger la communauté.",
"L'incendie de forêt franchit l'autoroute et brûle le lotissement.",
]},
"drought": {
"en": [
"{A severe drought|Several years of drought|A prolonged drought|Dry conditions|The drought} {devastated|reduced|ruined|hit} {crops|wheat yields|the harvest|farms} {across the Prairies|in <PLACE>|in the region}.",
"{Crops failed|Crop yields fell sharply|The harvest was lost|Wheat yields were cut in half} because of {the drought|lack of rain|heat and dry weather}.",
"{Topsoil|Dust|Soil} blew away in {dust storms|the wind|black blizzards} and fields turned to dust.",
"{Farmers|Ranchers|Farm families} {sold their cattle|abandoned their farms|ran out of feed|lost their livestock} because of the drought.",
"A {heat wave|record heat wave|prolonged heat wave} with temperatures {above|over|near} <N> C {caused|led to} {deaths|dozens of deaths|heat stroke|hospital admissions} {among the elderly|in the city|}.",
"{Wells|Dugouts|Reservoirs|Streams} {dried up|ran dry|fell to record lows} and water had to be hauled.",
"{Extreme heat|Hot, dry weather|The heat} {scorched|withered|burned up} {pastures|crops|gardens}.",
"Grasshoppers and drought destroyed the crops.",
"{Rainfall|Precipitation} was {the lowest on record|less than half of normal|far below normal}.",
"Water restrictions were imposed as the drought continued.",
"The heat wave lasted {a week|several days|two weeks}, with hot nights offering no relief.",
"Feed shortages forced the sale of livestock.",
],
"fr": [
"{Une grave sécheresse|Plusieurs années de sécheresse|Une sécheresse prolongée|Le temps sec|La sécheresse} {dévaste|réduit|ruine|frappe} {les récoltes|les rendements de blé|la récolte|les fermes} {dans les Prairies|dans <PLACE>|dans la région}.",
"{Les récoltes sont perdues|Les rendements chutent|La récolte est perdue|Les rendements de blé baissent de moitié} à cause {de la sécheresse|du manque de pluie|de la chaleur et du temps sec}.",
"{La terre arable|La poussière|Le sol} est emporté{e|} par {des tempêtes de poussière|le vent} et les champs se transforment en poussière.",
"{Les agriculteurs|Les éleveurs|Des familles agricoles} {vendent leur bétail|abandonnent leurs fermes|manquent de fourrage|perdent leur bétail} à cause de la sécheresse.",
"Une {vague de chaleur|canicule|vague de chaleur record|canicule prolongée} avec des températures {de plus de|supérieures à|près de} <N> C {cause|entraîne} {des décès|des dizaines de décès|des coups de chaleur|des hospitalisations} {chez les personnes âgées|dans la ville|}.",
"{Les puits|Les étangs-réservoirs|Les réservoirs|Les ruisseaux} {s'assèchent|tarissent|atteignent des niveaux records} et l'eau doit être transportée.",
"{La chaleur extrême|Le temps chaud et sec|La chaleur} {brûle|dessèche|grille} {les pâturages|les cultures|les jardins}.",
"Les sauterelles et la sécheresse détruisent les récoltes.",
"{Les précipitations|La pluie} {sont les plus faibles jamais enregistrées|atteignent moins de la moitié de la normale|sont bien inférieures à la normale}.",
"Des restrictions d'eau sont imposées alors que la sécheresse se poursuit.",
"La canicule dure {une semaine|plusieurs jours|deux semaines} et les nuits chaudes n'apportent aucun répit.",
"La pénurie de fourrage force la vente du bétail.",
]},
"ground": {
"en": [
"A {landslide|rock slide|rockslide|mudslide|slide of clay|debris flow} {buried|swept away|destroyed|struck} {part of <PLACE>|several houses|the highway|a section of railway}.",
"An earthquake of magnitude <N> {shook|struck|rocked} {the coast|<PLACE>|the region}, {cracking buildings|damaging chimneys|toppling chimneys|breaking windows}.",
"A {snow avalanche|avalanche|series of avalanches} {buried|swept over|struck} {a work camp|skiers|the highway|a mining camp|a rail line} {in the mountains|in the pass|in the backcountry}.",
"A tsunami {triggered|caused} by an {offshore|undersea} earthquake {struck|hit|swept} the coast, {washing away houses|destroying boats|drowning residents}.",
"The {slide|landslide|avalanche} {killed|buried|trapped} {<N> people|several workers|residents}.",
"{Rescuers|Rescue teams|Search teams} {dug through|searched} the {debris|snow|rubble} for survivors.",
"{Tremors|Aftershocks|The quake} {were felt|was felt} {hundreds of kilometres away|across the province|for days}.",
"{Saturated soil|Heavy rain|Spring thaw} {triggered|caused|set off} a {landslide|mudslide|slide} that {blocked the river|cut the highway|carried houses into the river}.",
"A {sinkhole|ground collapse|land subsidence} {swallowed|damaged|destroyed} {homes|a road|several buildings}.",
"{Rock fall|Falling rocks} {blocked|closed|buried} the {highway|rail line|road} in the canyon.",
"The mountainside gave way and millions of tonnes of rock fell into the valley.",
"Avalanche control was carried out after the slide closed the pass.",
],
"fr": [
"Un {glissement de terrain|éboulement|éboulement de roches|glissement de boue|glissement d'argile|coulée de débris} {ensevelit|emporte|détruit|frappe} {une partie de <PLACE>|plusieurs maisons|l'autoroute|un tronçon de voie ferrée}.",
"Un {séisme|tremblement de terre} de magnitude <N> {secoue|frappe|ébranle} {la côte|<PLACE>|la région} et {fissure des bâtiments|endommage des cheminées|fait tomber des cheminées|brise des fenêtres}.",
"Une {avalanche|avalanche de neige|série d'avalanches} {ensevelit|emporte|frappe} {un camp de travail|des skieurs|l'autoroute|un camp minier|une voie ferrée} {dans les montagnes|dans le col|dans l'arrière-pays}.",
"Un {tsunami|raz-de-marée} causé par un séisme {au large|sous-marin} {frappe|balaie} la côte et {emporte des maisons|détruit des bateaux|noie des résidents}.",
"Le {glissement|glissement de terrain|l'avalanche} {tue|ensevelit|piège} {<N> personnes|plusieurs ouvriers|des résidents}.",
"{Les sauveteurs|Les équipes de secours|Les équipes de recherche} {fouillent|creusent dans} {les débris|la neige|les décombres} à la recherche de survivants.",
"{Les secousses|Les répliques|Le séisme} {sont ressenties|est ressenti} {à des centaines de kilomètres|dans toute la province|pendant des jours}.",
"{Le sol saturé|De fortes pluies|Le dégel printanier} {déclenchent|provoquent} un {glissement de terrain|glissement de boue|éboulement} qui {bloque la rivière|coupe l'autoroute|emporte des maisons dans la rivière}.",
"Un {affaissement de terrain|effondrement du sol|gouffre} {engloutit|endommage|détruit} {des maisons|une route|plusieurs bâtiments}.",
"Des chutes de pierres {bloquent|ferment|ensevelissent} {l'autoroute|la voie ferrée|la route} dans le canyon.",
"Le flanc de la montagne cède et des millions de tonnes de roc tombent dans la vallée.",
"Des travaux de déclenchement préventif d'avalanches sont effectués après la fermeture du col.",
]},
"industrial": {
"en": [
"An explosion {in the coal mine|at the colliery|in the mine|at the refinery|at the munitions plant|at the chemical plant} {killed|trapped|injured} {<N> miners|<N> workers|dozens of workers}.",
"A {fire|blaze|major fire} {broke out|started} in {a hotel|a warehouse|a factory|a sawmill|an apartment building|a hospital|a school|a church} and {spread to|destroyed|levelled} {neighbouring buildings|the downtown area|a city block}.",
"A {chemical leak|toxic gas leak|chlorine leak|propane explosion|gas explosion} {forced the evacuation of|endangered|killed} {<N> residents|thousands of people|nearby homes}.",
"A {pipeline|tanker truck|storage tank|tailings dam} {ruptured|leaked|failed|exploded}, spilling {oil|chemicals|toxic waste|gasoline} into {the river|the lake|the town}.",
"{A bridge|A dam|A building|The roof of the arena|A grain elevator} {collapsed|gave way|failed}, {killing|injuring|trapping} {<N> people|several workers}.",
"{Methane|Coal dust|Firedamp|Gas} {exploded|ignited} {underground|in the mine}, {trapping|killing} the miners.",
"The {fire|blaze} {destroyed|gutted|razed} {much of the town|the business district|hundreds of buildings|the factory}.",
"{A power failure|An equipment failure|A blackout|A transformer fire} {left|cut power to} {<N> customers|the city} {for days|without electricity}.",
"Firefighters {battled the blaze|fought the fire} for {hours|days} {at the plant|downtown|in the industrial park}.",
"A {sulphur|chemical|ammonia|chlorine} cloud drifted over {<PLACE>|the town|the city}, forcing an evacuation.",
"{Faulty wiring|A gas leak|An overheated furnace|Careless smoking} was blamed for the fire.",
"Contaminated water from the spill {killed fish|polluted wells|forced a ban on drinking water}.",
],
"fr": [
"Une explosion {dans la mine de charbon|dans la houillère|dans la mine|à la raffinerie|à l'usine de munitions|à l'usine chimique} {tue|emprisonne|blesse} {<N> mineurs|<N> ouvriers|des dizaines de travailleurs}.",
"Un {incendie|brasier|important incendie} {se déclare|éclate} dans {un hôtel|un entrepôt|une usine|une scierie|un immeuble d'appartements|un hôpital|une école|une église} et {se propage aux|détruit|rase} {édifices voisins|centre-ville|pâté de maisons}.",
"Une {fuite de produits chimiques|fuite de gaz toxique|fuite de chlore|explosion de propane|explosion de gaz} {force l'évacuation de|menace|tue} {<N> résidents|milliers de personnes|maisons voisines}.",
"{Un oléoduc|Un camion-citerne|Un réservoir|Une digue de résidus miniers} {se rompt|fuit|cède|explose} et déverse {du pétrole|des produits chimiques|des déchets toxiques|de l'essence} dans {la rivière|le lac|la ville}.",
"{Un pont|Un barrage|Un édifice|Le toit de l'aréna|Un silo à grains} {s'effondre|cède}, {tuant|blessant|piégeant} {<N> personnes|plusieurs ouvriers}.",
"{Le méthane|La poussière de charbon|Le grisou|Le gaz} {explose|s'enflamme} {sous terre|dans la mine}, {emprisonnant|tuant} les mineurs.",
"L'incendie {détruit|ravage|rase} {une grande partie de la ville|le quartier des affaires|des centaines de bâtiments|l'usine}.",
"{Une panne de courant|Une défaillance d'équipement|Une panne générale|Un incendie de transformateur} prive {<N> abonnés|la ville} d'électricité {pendant des jours|}.",
"Les pompiers combattent l'incendie pendant {des heures|des jours} {à l'usine|au centre-ville|dans le parc industriel}.",
"Un nuage {de soufre|chimique|d'ammoniac|de chlore} se répand sur {<PLACE>|la ville}, forçant une évacuation.",
"{Un câblage défectueux|Une fuite de gaz|Une fournaise surchauffée|Une cigarette mal éteinte} est à l'origine de l'incendie.",
"L'eau contaminée par le déversement {tue les poissons|pollue les puits|entraîne une interdiction de boire l'eau}.",
]},
"transport": {
"en": [
"A {passenger train|freight train|train|express train} {derailed|left the tracks|jumped the rails} {near <PLACE>|on a curve|on a bridge|in the canyon}, {killing|injuring} {<N> passengers|<N> people|the crew}.",
"{Two trains|A passenger train and a freight train|Two freight trains} collided {head-on|near <PLACE>|on the main line}, killing {<N> people|the engineers|<N> passengers}.",
"{An airliner|A passenger plane|A chartered aircraft|A small plane|A military transport plane} crashed {shortly after takeoff|on approach to the airport|into a mountain|into the sea|in a forest}, killing {all <N> people on board|<N> passengers and crew|everyone on board}.",
"The {steamship|steamer|ship|freighter|ferry|fishing vessel|schooner|tanker} {sank|ran aground|struck rocks|capsized|foundered|went down} {in a storm|in fog|off the coast|in the gulf|on the lake}, with the loss of <N> lives.",
"{A bus|A school bus|A tour bus|A truck} {collided with|was struck by|plunged off} {a train at a level crossing|a truck|the road into a ravine|the bridge}, killing <N> people.",
"{Passengers|Crew members|Fishermen|Sailors} {drowned|were lost|died} when the {vessel|boat|ship} {sank|capsized|went down}.",
"A {multi-vehicle pileup|chain-reaction collision|highway crash} involving {<N> vehicles|dozens of cars} {killed|injured} {several people|<N> people}.",
"{Investigators|The Transportation Safety Board|An inquiry} {blamed|found|pointed to} {pilot error|a broken rail|excessive speed|signal failure|human error} as the cause of the {crash|accident|derailment|collision}.",
"The {wreckage|wreck|debris} was {found|scattered} {in the forest|on the seabed|over a wide area}.",
"{Two ships|A ferry and a freighter|Two vessels} collided {in the harbour|in fog|in the strait}.",
"{Rescuers|The coast guard|Search and rescue crews} {pulled survivors from the water|searched for survivors|recovered the bodies}.",
"The {plane|aircraft|helicopter} {went missing|disappeared from radar|lost power} {in bad weather|over the mountains|over the lake}.",
],
"fr": [
"Un {train de passagers|train de marchandises|train|train express} {déraille|quitte les rails|sort des voies} {près de <PLACE>|dans une courbe|sur un pont|dans le canyon} et {tue|blesse} {<N> passagers|<N> personnes|l'équipage}.",
"{Deux trains|Un train de passagers et un train de marchandises|Deux trains de marchandises} entrent en collision {de plein fouet|près de <PLACE>|sur la voie principale}, tuant {<N> personnes|les mécaniciens|<N> passagers}.",
"{Un avion de ligne|Un avion de passagers|Un avion nolisé|Un petit avion|Un avion de transport militaire} s'écrase {peu après le décollage|à l'approche de l'aéroport|contre une montagne|en mer|en forêt} et tue {les <N> personnes à bord|<N> passagers et membres d'équipage|toutes les personnes à bord}.",
"Le {vapeur|navire|cargo|traversier|bateau de pêche|goélette|pétrolier} {coule|s'échoue|heurte des rochers|chavire|sombre} {dans une tempête|dans le brouillard|au large des côtes|dans le golfe|sur le lac}; <N> personnes périssent.",
"{Un autobus|Un autobus scolaire|Un autocar|Un camion} {entre en collision avec|est happé par|plonge} {un train à un passage à niveau|un camion|dans un ravin|du haut du pont}, tuant <N> personnes.",
"{Des passagers|Des membres d'équipage|Des pêcheurs|Des marins} {se noient|disparaissent|périssent} lorsque le {navire|bateau|vapeur} {coule|chavire|sombre}.",
"Un {carambolage|accident en chaîne|accident de la route} impliquant {<N> véhicules|des dizaines de voitures} {tue|blesse} {plusieurs personnes|<N> personnes}.",
"{Les enquêteurs|Le Bureau de la sécurité des transports|Une enquête} {attribuent|attribue} {l'écrasement|l'accident|le déraillement|la collision} à {une erreur de pilotage|un rail brisé|une vitesse excessive|une défaillance de signalisation|une erreur humaine}.",
"{L'épave|Les débris} {sont retrouvés|est retrouvée} {en forêt|au fond de l'eau|sur une grande superficie}.",
"{Deux navires|Un traversier et un cargo|Deux bateaux} entrent en collision {dans le port|dans le brouillard|dans le détroit}.",
"{Les sauveteurs|La garde côtière|Les équipes de recherche} {repêchent des survivants|recherchent des survivants|récupèrent les corps}.",
"{L'avion|L'appareil|L'hélicoptère} {disparaît|disparaît des écrans radar|perd de la puissance} {par mauvais temps|au-dessus des montagnes|au-dessus du lac}.",
]},
"epidemic": {
"en": [
"{An epidemic|An outbreak|A severe outbreak|A pandemic|A wave} of {influenza|Spanish flu|smallpox|typhoid|cholera|polio|diphtheria|measles|scarlet fever|tuberculosis|SARS|meningitis|E. coli infection|whooping cough} {spread through|swept|hit|broke out in} {<PLACE>|the city|the province|the country}.",
"There were {<N> cases|<N> cases and <N> deaths|thousands of cases} {reported|recorded|confirmed}.",
"{Schools|Churches|Theatres|Schools, churches and theatres} were closed to {stop|slow|contain} the spread of the {disease|epidemic|virus|infection}.",
"{Hospitals|Emergency hospitals|Clinics} were {overwhelmed|set up|filled to capacity} {with patients|as cases rose}.",
"{Vaccination|Mass vaccination|Quarantine|Isolation} was {ordered|made compulsory|imposed|introduced} to {control|stop|contain} the {outbreak|epidemic}.",
"{Contaminated water|A contaminated water supply|Tainted food} {caused|was the source of} the outbreak; {<N>|thousands of} people fell ill.",
"The {disease|virus|illness|infection} {killed|claimed|took the lives of} {<N> people|many children|hundreds of residents}.",
"{Iron lungs|Nurses|Doctors|Medical teams} were {flown in|sent|brought in} to {help|treat} the {sick|patients|infected}.",
"{Health officials|The board of health|Public health authorities} {declared|announced} {an epidemic|a public health emergency|a quarantine}.",
"The {epidemic|outbreak|pandemic} {peaked|subsided|ended} {in the fall|after several months|the following year}.",
"Children were {most affected|hardest hit|paralysed by the disease}.",
"The infection spread {from person to person|through the water supply|among workers in the camps}.",
],
"fr": [
"{Une épidémie|Une éclosion|Une grave éclosion|Une pandémie|Une vague} de {grippe|grippe espagnole|variole|typhoïde|choléra|poliomyélite|diphtérie|rougeole|scarlatine|tuberculose|SRAS|méningite|E. coli|coqueluche} {se propage dans|frappe|touche|éclate dans} {<PLACE>|la ville|la province|le pays}.",
"On {compte|signale|confirme} {<N> cas|<N> cas et <N> décès|des milliers de cas}.",
"{Les écoles|Les églises|Les théâtres|Les écoles, les églises et les théâtres} sont fermés pour {freiner|ralentir|contenir} la propagation de la {maladie|épidémie|infection}.",
"{Les hôpitaux|Des hôpitaux d'urgence|Les cliniques} sont {débordés|aménagés|remplis} {de malades|alors que les cas augmentent}.",
"{La vaccination|La vaccination de masse|La quarantaine|L'isolement} est {ordonnée|rendue obligatoire|imposée} pour {contrôler|freiner|contenir} {l'éclosion|l'épidémie}.",
"{De l'eau contaminée|Un réseau d'eau contaminé|Des aliments contaminés} {cause|est à l'origine de} l'éclosion; {<N>|des milliers de} personnes tombent malades.",
"La {maladie|virus|infection} {tue|emporte|coûte la vie à} {<N> personnes|de nombreux enfants|des centaines de résidents}.",
"{Des poumons d'acier|Des infirmières|Des médecins|Des équipes médicales} sont {envoyés|dépêchés|appelés} pour {soigner|aider} les {malades|patients|personnes infectées}.",
"{Les autorités sanitaires|Le bureau de santé|La santé publique} {déclarent|annoncent} {une épidémie|une urgence de santé publique|une quarantaine}.",
"L'{épidémie|éclosion|pandémie} {atteint son sommet|s'atténue|prend fin} {à l'automne|après plusieurs mois|l'année suivante}.",
"Les enfants sont {les plus touchés|durement frappés|paralysés par la maladie}.",
"L'infection se propage {de personne à personne|par le réseau d'eau|parmi les travailleurs des camps}.",
]},
}

GENERIC = {
"en": ["{<N>|Several|Many|Dozens of} people were {killed|injured|hospitalized}.", "Damage was estimated at ${<N>|several} million.",
       "A state of emergency was declared.", "{The army|Soldiers|The military} {helped|assisted} with the response.",
       "The Red Cross opened shelters.", "{<N>|Hundreds of|Thousands of} residents were evacuated.",
       "Power was out for {days|several days|a week}.", "Insured losses reached ${<N>|several} million.",
       "{Roads|Several roads|The highway} {were|was} closed.", "Emergency crews worked through the night.",
       "The provincial government provided disaster assistance.", "{<N>|Several} {homes|buildings} were destroyed.",
       "The death toll rose to <N>.", "Local authorities asked for federal help."],
"fr": ["{<N>|Plusieurs|De nombreuses|Des dizaines de} personnes sont {tuées|blessées|hospitalisées}.", "Les dommages sont évalués à {<N>|plusieurs} millions de dollars.",
       "L'état d'urgence est déclaré.", "{L'armée|Des soldats|Les militaires} {aident|participent} aux opérations.",
       "La Croix-Rouge ouvre des refuges.", "{<N>|Des centaines de|Des milliers de} résidents sont évacués.",
       "Le courant est coupé pendant {des jours|plusieurs jours|une semaine}.", "Les pertes assurées atteignent {<N>|plusieurs} millions de dollars.",
       "{Des routes|Plusieurs routes|L'autoroute} {sont fermées|est fermée}.", "Les équipes d'urgence travaillent toute la nuit.",
       "Le gouvernement provincial offre une aide aux sinistrés.", "{<N>|Plusieurs} {maisons|bâtiments} sont détruits.",
       "Le bilan s'élève à <N> morts.", "Les autorités locales demandent l'aide fédérale."],
}

def expand(t, rng, lang):
    while '{' in t:
        t = re.sub(r'\{([^{}]*)\}', lambda m: rng.choice(m.group(1).split('|')), t, count=1)
    t = t.replace('<N>', str(rng.choice([rng.randint(2, 99), rng.randint(100, 9999)])))
    t = t.replace('<RIVER>', rng.choice(RIVERS))
    t = t.replace('<PLACE>', rng.choice(PLACES_EN if lang == 'en' else PLACES_FR))
    return t

def generate(n_per_class_lang=300, seed=0, classes=None):
    rng = random.Random(seed)
    out = []
    for c, d in H.items():
        for lang in ('en', 'fr'):
            for _ in range(n_per_class_lang):
                k = rng.choice([1, 2, 2, 3, 3, 4])
                parts = [expand(t, rng, lang) for t in rng.sample(d[lang], k)]
                parts += [expand(t, rng, lang) for t in rng.sample(GENERIC[lang], rng.randint(0, 3))]
                rng.shuffle(parts)
                out.append((c, ' '.join(parts)))
    return out

if __name__ == '__main__':
    for c, t in generate(1, seed=3)[:18]: print(c, '|', t)
