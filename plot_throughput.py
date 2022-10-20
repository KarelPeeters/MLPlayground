import matplotlib.pyplot as plt
import numpy as np

NaN = np.nan
nan = np.nan
false = False
true = True


def data(version: int):
    # @formatter:off
    if version == -1:
        name = "pytorch"
        size_inert = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        size_op = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        caches = [False]
        throughput = [0.005929777877025637, 0.012503924431705992, 0.020104018306516522, 0.05084093807475824, 0.10642795708018347, 0.1878051036640892, 0.35721483899475254, 0.6861120556891752, 0.9308538524302519, 1.91675270075054, 3.9277426075484394, 5.037202764739247, 5.375030960178387, 5.529579801480131, 5.631767131385054, 5.915558012221062, 0.012073355062745131, 0.012689095046452922, 0.0283536291483939, 0.09150149353862616, 0.13206041908277788, 0.26332773724673764, 0.5274109210549035, 0.9049216618728657, 3.165970186995953, 6.8333135076127824, 8.226178041342768, 8.838709531438159, 9.061585142364114, 8.636948561927374, 9.658667332625319, 10.025196124916825, 0.03249648401561536, 0.05701065975646639, 0.1190753298047519, 0.19247192238075672, 0.41382011397226404, 0.9870808333603838, 1.7511162315306117, 3.734407504283089, 8.092165230361754, 10.122336125045617, 10.668267077421273, 10.253432028768538, 10.62292130675851, 11.53686554623086, 11.611816779279591, 11.8911900931356, 0.04486506792775195, 0.057555142135889276, 0.17610088014149144, 0.3907500400128438, 0.7431168121605519, 1.7615780492381974, 1.8801742395070558, 3.68341794782848, 9.313723152634411, 10.289895158316137, 9.766406312504797, 9.310348936981626, 10.605414768438862, 10.28191869486423, 10.616710260294585, 10.567675223045566, 0.09350664931915653, 0.11290094902407243, 0.2650752043377757, 0.4746050314148021, 0.9704756765565862, 2.653241012432387, 5.702354953987026, 10.100559554839581, 10.837809493157955, 11.154786313559395, 10.312959545477288, 10.892722197590153, 11.091715890222865, 11.342946828620954, 11.368183637091198, 11.366060434343543, 0.1688180587978184, 0.2129420581729679, 0.910293158090952, 0.7552547361843229, 3.0846088972564663, 3.426198478745301, 12.825206188278552, 11.304507622673983, 11.457227328989044, 10.184328192827689, 11.4744250670474, 11.435476103697274, 11.663315718305372, 11.766717021939913, 11.862746316878232, 11.86322817799512, 0.2742413562634953, 0.8281568012209811, 1.029069755188795, 2.9468499541334294, 5.786282677220224, 10.677715454089533, 14.322668407079195, 11.954712107579882, 11.696865165275836, 10.884815316903198, 11.87803745173724, 11.885893897192538, 11.949009656520477, 11.994108877530865, 12.022460649524488, nan, 0.5170192478736061, 1.0674774166182506, 2.0944051969657953, 4.05562684806758, 7.74078932766912, 11.370715150667394, 15.072269724658872, 12.143392284806584, 10.757996375673802, 11.737168551749846, 11.86860448378782, 11.819231193995796, 11.849423966322616, 11.763955203611477, nan, nan, 0.5266009477671478, 1.113038874654575, 2.054675270573009, 4.174057531202236, 7.951751064644593, 12.947206883523158, 16.194932048657183, 11.585797184410264, 11.860266702291721, 11.884479060498585, 11.802366818316072, 11.793954890481352, 10.820103700739475, nan, nan, nan, 0.6158666483358741, 1.1962107106984772, 2.379584641026131, 4.720361266996374, 8.283113369183013, 12.409973091207576, 15.052348455363932, 12.060542766877628, 11.815338918743404, 11.939479637360746, 11.976893601972167, 11.992705556772167, nan, nan, nan, nan, 0.5949121060672666, 1.182347763550423, 2.4424015286228826, 4.893615155455969, 7.854079496856071, 12.138166123136859, 15.84136131281047, 12.08480010580498, 11.870002016950691, 11.97831445950255, 11.996459988614877, nan, nan, nan, nan, nan, 0.6098111173971869, 1.2460095795609383, 2.3983881603588277, 4.980117776199283, 8.065165504938264, 11.522759385610462, 16.13756121945237, 11.830296385835025, 11.517742714083273, 11.478408974977189, nan, nan, nan, nan, nan, nan, 0.5961672531298948, 1.2022426885178013, 2.509269442060447, 4.943171716136275, 8.060505637453234, 11.410610736051687, 16.059274912175123, 12.04524386318916, 11.758329824255764, nan, nan, nan, nan, nan, nan, nan, 0.598592768738838, 1.238598944436256, 2.48986604600349, 4.914310156660365, 7.939355014420499, 11.266588022877785, 15.714551335101897, 11.816956292387436, nan, nan, nan, nan, nan, nan, nan, nan, 0.614543614150464, 1.2101792924188106, 2.4001624583562338, 4.66774958457027, 7.9348605083262, 11.263514978988164, 15.87190030516582, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0.604866322800646, 1.2572081273657643, 2.5136282893138744, 4.926125846357704, 7.915526514227537, 11.198528817913305, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
    elif version == 0:
        name = "orig"
        size_inert = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        size_op = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        caches = [false, true]
        throughput = [0.06344159, 0.07114764, 0.1353636, 0.1475947, 0.15573956, 0.29172206, 0.49049243, 0.54945284, 0.75832874, 0.94190055, 0.83950204, 1.269805, 0.9712342, 1.4446108, 1.0291744, NaN, 0.62855864, NaN, 0.619615, NaN, 0.695108, NaN, 0.7743598, NaN, 0.8332326, NaN, 0.83519375, NaN, 0.099766746, 0.1621101, 0.31650725, 0.3094095, 0.5610377, 0.5539465, 1.140104, 1.1569225, 1.0620928, 2.0398579, 2.2898443, 3.3113692, 2.4933965, 3.4413767, 1.6373777, NaN, 1.5991051, NaN, 1.4590876, NaN, 1.5929617, NaN, 1.6278611, NaN, 1.6307456, NaN, 1.6175984, NaN, 0.31490198, 0.3521068, 0.5903788, 0.39008275, 0.74468565, 1.5619665, 2.212496, 2.2424622, 3.798894, 4.4217095, 4.349117, 5.9278617, 2.9903243, 4.76456, 2.9334798, NaN, 2.994456, NaN, 3.0186732, NaN, 2.995655, NaN, 3.040011, NaN, 3.0363057, NaN, 3.0493426, NaN, 0.39736432, 0.5744473, 1.0508577, 1.1524487, 2.303116, 2.278465, 3.9499435, 4.182782, 7.2866316, 8.969849, 5.051241, 7.5092463, 5.0119524, 7.8933477, 5.17205, NaN, 5.24285, NaN, 5.214327, NaN, 5.3051534, NaN, 5.3022594, NaN, 5.305993, NaN, 5.291925, NaN, 1.149781, 0.65614974, 1.9581027, 2.136367, 4.5292287, 4.6026754, 4.4037423, 9.2697735, 7.629394, 10.610528, 7.7863674, 11.415781, 8.444828, 8.462812, 8.39576, NaN, 8.151586, NaN, 8.351279, NaN, 8.472351, NaN, 8.435933, NaN, 8.467544, NaN, 8.313502, NaN, 2.2374115, 2.2441509, 4.9057317, 4.08811, 9.24103, 8.570043, 9.687874, 12.127089, 9.496857, 13.519625, 10.036564, 11.7288685, 10.565858, 10.5204, 9.791821, NaN, 9.57072, NaN, 9.85799, NaN, 9.779582, NaN, 9.833611, NaN, 9.819988, NaN, 9.7547, NaN, 3.6322148, 3.8704317, 8.49069, 8.669766, 11.235559, 11.219698, 10.51924, 12.90493, 10.228167, 14.343123, 10.508228, 13.93955, 10.719356, 11.827418, 9.787298, NaN, 9.675035, NaN, 9.725634, NaN, 9.773756, NaN, 9.758115, NaN, 9.758244, NaN, NaN, NaN, 7.4599056, 7.822132, 9.831694, 10.123931, 12.020094, 12.866626, 10.575231, 14.680947, 10.188829, 13.935477, 10.47332, 14.950217, 10.47562, 12.673833, 9.343737, NaN, 9.383268, NaN, 9.438363, NaN, 9.473472, NaN, 9.36444, NaN, NaN, NaN, NaN, NaN, 8.87304, 9.567359, 12.270643, 11.782485, 14.575489, 14.515591, 11.698655, 15.446619, 11.15213, 15.302861, 11.059339, 14.861975, 11.013438, 12.5340185, 10.016043, NaN, 9.946394, NaN, 9.928379, NaN, 9.945244, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 10.725082, 11.035342, 13.587039, 13.979396, 15.4353695, 14.71152, 11.846886, 16.170137, 11.476744, 16.13388, 11.199854, 15.13454, 11.073624, 12.757491, 10.119047, NaN, 10.089806, NaN, 10.052855, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 11.644374, 12.315009, 14.866318, 14.64825, 15.941067, 15.728765, 11.895274, 16.246582, 11.627516, 16.21895, 11.247279, 15.601475, 11.103438, 12.943215, 10.101713, NaN, 10.0704155, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 12.107052, 11.650776, 14.198448, 14.166811, 14.874144, 15.300714, 11.895558, 16.256273, 11.650422, 16.584105, 11.22688, 15.587927, 11.084756, 12.992922, 10.0708065, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 11.788665, 11.851682, 13.726862, 13.944646, 14.545201, 15.531681, 11.907858, 16.428942, 11.7212105, 16.852392, 11.398789, 15.798792, 11.264476, 13.079989, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 11.227949, 11.117514, 13.714409, 14.069106, 14.898033, 15.558524, 11.921279, 16.436108, 11.773223, 16.907642, 11.414713, 15.68514, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 10.808497, 10.806891, 13.510347, 13.82525, 14.713649, 15.589999, 11.960232, 16.378725, 11.749455, 16.554657, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 10.690218, 10.944516, 13.436084, 13.930705, 14.771353, 15.619802, 11.954219, 16.389666, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
    elif version == 1:
        name = "cache?"
        size_inert = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        size_op = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        caches = [false, true]
        throughput = [0.06793017, 0.08007933, 0.12953025, 0.062171068, 0.2655232, 0.2371286, 0.53562766, 0.57936084, 0.7884213, 0.8944274, 0.83620435, 1.0068352, 0.7289304, 1.189951, 0.85595816, NaN, 0.66400766, NaN, 0.61653864, NaN, 0.71153814, NaN, 0.7866166, NaN, 0.8280526, NaN, 0.8293514, NaN, 0.14506583, 0.11207252, 0.2878895, 0.3028691, 0.60085326, 0.5020607, 1.1145221, 0.49439818, 1.5135765, 2.1211617, 2.1486895, 2.616534, 2.4298673, 3.5574245, 1.6078943, NaN, 1.5543293, NaN, 1.5869974, NaN, 1.6039191, NaN, 1.60631, NaN, 1.6182555, NaN, 1.6203105, NaN, 0.29730967, 0.33530968, 0.56061554, 0.3681117, 1.0651295, 1.1650634, 1.4551916, 2.273251, 3.5478957, 4.257475, 4.1871896, 6.15115, 2.9746547, 4.6448193, 2.8040996, NaN, 2.926279, NaN, 3.0145226, NaN, 3.0020914, NaN, 2.9982808, NaN, 3.0298278, NaN, 3.0310135, NaN, 0.56572366, 0.5704885, 1.0997167, 1.4191583, 1.9988145, 1.9227303, 4.4348693, 4.4950714, 6.4858155, 8.588566, 5.040562, 7.0831423, 4.9526086, 8.012724, 4.9387593, NaN, 5.163649, NaN, 5.2175364, NaN, 5.265101, NaN, 5.266864, NaN, 5.2757416, NaN, 5.26382, NaN, 1.0989057, 1.0868827, 2.337437, 2.4793947, 4.546502, 5.265428, 7.6760654, 9.2986965, 7.5929484, 10.992097, 7.7283173, 11.151478, 8.2262945, 8.337043, 8.144797, NaN, 8.258708, NaN, 8.339231, NaN, 8.515947, NaN, 8.394697, NaN, 8.440786, NaN, 8.440687, NaN, 2.1241856, 2.1502397, 4.448108, 4.983666, 4.4086275, 6.7617292, 9.276987, 8.359697, 9.455427, 13.432033, 9.180538, 11.503911, 10.097669, 10.451226, 9.645496, NaN, 9.703645, NaN, 9.860794, NaN, 9.79358, NaN, 9.828591, NaN, 9.827681, NaN, 9.842066, NaN, 3.9842677, 5.1830125, 9.198248, 8.261211, 10.366025, 10.619981, 9.988211, 12.814758, 9.711552, 14.242448, 9.929971, 12.825098, 10.609644, 11.291431, 9.765625, NaN, 9.741185, NaN, 9.684491, NaN, 9.738653, NaN, 9.763375, NaN, 9.74595, NaN, NaN, NaN, 6.6560183, 7.2955503, 9.695753, 9.625296, 11.701525, 12.295956, 10.766249, 14.6741705, 10.1568165, 14.156821, 10.2910795, 14.840293, 10.325062, 12.593085, 9.586232, NaN, 9.506382, NaN, 9.473596, NaN, 9.47264, NaN, 9.459185, NaN, NaN, NaN, NaN, NaN, 7.9340625, 8.153851, 10.0577345, 11.593416, 13.596725, 14.04321, 11.567401, 15.65582, 10.846145, 15.272228, 10.836285, 15.391463, 10.998832, 12.47428, 9.995407, NaN, 9.990663, NaN, 9.921568, NaN, 9.945183, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 9.168183, 9.773255, 13.163207, 13.5234585, 15.261231, 14.827025, 11.68647, 16.065287, 11.330503, 15.890599, 11.118728, 15.401556, 11.048046, 12.632703, 10.133932, NaN, 10.088839, NaN, 10.050567, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 9.848955, 9.774258, 13.552286, 13.847457, 14.9420185, 14.9046545, 11.764862, 15.778861, 11.416635, 16.504547, 11.1946335, 15.491157, 11.125317, 12.753733, 10.071992, NaN, 10.065608, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 10.285529, 10.320592, 13.389131, 12.540508, 14.004286, 14.627467, 11.617159, 15.9874935, 11.476225, 16.637632, 11.34934, 15.613088, 11.217265, 12.858913, 10.093786, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 9.890314, 9.837525, 12.804006, 12.720959, 13.995807, 15.0844145, 11.6762295, 15.749139, 11.433166, 16.542934, 11.357987, 15.632644, 11.218341, 12.86287, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 9.448279, 10.065969, 12.776235, 13.050401, 14.117135, 14.947361, 11.796216, 16.156271, 11.605649, 16.738745, 11.432128, 15.616514, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 9.058403, 9.242766, 12.371924, 12.70581, 14.177243, 14.939238, 11.86636, 16.168789, 11.650664, 16.830366, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 8.960367, 9.291562, 12.40378, 12.8497505, 14.1791525, 15.041811, 11.874607, 16.176846, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
    elif version == 2:
        name = "block reduce"
        size_inert = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        size_op = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        caches = [false, true]
        throughput = [0.07034159, 0.08191051, 0.12126596, 0.13266703, 0.29104543, 0.27863052, 0.4472137, 0.5414666, 0.68291295, 0.8237094, 0.7905125, 1.0845095, 0.85860914, 1.263078, 0.8846044, NaN, 0.62707084, NaN, 0.63133824, NaN, 0.7199064, NaN, 0.7810984, NaN, 0.82109964, NaN, 0.830029, NaN, 0.1509437, 0.15771762, 0.23956852, 0.31147912, 0.5458301, 0.56315804, 1.0449622, 1.3151951, 1.6483586, 2.119653, 2.1158907, 2.9877014, 2.426899, 3.224487, 1.6065942, NaN, 1.5693176, NaN, 1.6025447, NaN, 1.5913801, NaN, 1.617343, NaN, 1.6141299, NaN, 1.6196604, NaN, 0.24284813, 0.3380481, 0.5839013, 0.64563084, 0.6226979, 0.9987374, 2.2059453, 1.5267583, 3.3466954, 2.9922009, 4.3396173, 5.919031, 2.9591484, 2.6532228, 2.9207222, NaN, 2.9623654, NaN, 3.0112386, NaN, 2.9823294, NaN, 3.0243092, NaN, 3.0300026, NaN, 3.0293946, NaN, 0.4930894, 0.57400465, 1.1533407, 1.2234123, 2.273251, 2.4328427, 3.6522453, 4.613363, 6.597083, 8.049243, 4.9603043, 4.6384935, 5.0014386, 3.5752954, 5.149708, NaN, 5.2382417, NaN, 5.263539, NaN, 5.2377024, NaN, 5.271267, NaN, 5.2762156, NaN, 5.271881, NaN, 1.087676, 1.1111978, 2.082622, 2.3784773, 4.543037, 3.7346268, 8.266941, 8.365564, 7.7660775, 7.448253, 7.490373, 6.31238, 8.2163725, 4.2900333, 8.075143, NaN, 7.9685354, NaN, 8.472587, NaN, 8.39285, NaN, 8.473785, NaN, 8.412574, NaN, 8.323757, NaN, 0.45513627, 1.8062015, 4.2757993, 5.233068, 9.003723, 4.8617167, 8.647755, 10.267812, 8.242647, 9.09127, 8.0289135, 7.597788, 8.363363, 4.7763724, 7.336803, NaN, 8.153415, NaN, 8.155811, NaN, 8.080038, NaN, 8.010304, NaN, 8.152244, NaN, 8.116776, NaN, 4.0685763, 4.0741386, 7.5496697, 8.137153, 8.336314, 9.159377, 8.744493, 9.515808, 8.706969, 11.512244, 8.744893, 7.969867, 8.57833, 4.6806674, 8.277076, NaN, 8.112234, NaN, 8.147189, NaN, 8.143157, NaN, 8.09575, NaN, 8.017104, NaN, NaN, NaN, 6.320747, 6.597083, 7.794004, 8.755732, 9.678042, 10.473032, 10.66392, 12.038302, 9.295525, 12.628939, 9.188721, 8.410938, 9.032718, 4.749906, 8.860519, NaN, 8.769772, NaN, 8.716319, NaN, 8.598463, NaN, 8.522387, NaN, NaN, NaN, NaN, NaN, 6.6190615, 7.7033467, 8.726888, 10.339053, 10.3021965, 12.380557, 10.918477, 13.39054, 9.664067, 12.720322, 9.391873, 8.432973, 9.453844, 4.828853, 9.143953, NaN, 9.028722, NaN, 9.024898, NaN, 8.8359165, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.2423625, 8.722099, 9.312316, 11.263427, 10.126081, 12.782125, 10.636267, 13.746161, 9.549038, 12.353895, 9.329911, 8.6045, 9.471332, 4.880064, 9.106136, NaN, 8.390969, NaN, 9.014174, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.5941577, 9.117346, 9.36122, 11.766495, 10.2899685, 12.698727, 10.565858, 13.491406, 9.32877, 12.844098, 9.389418, 8.816874, 9.405087, 4.9067616, 9.061234, NaN, 8.896342, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.7139397, 9.344709, 9.177005, 11.759251, 10.073537, 12.79777, 10.321432, 13.424706, 9.407944, 12.41728, 9.259452, 8.849214, 9.326952, 4.9079766, 8.921263, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.642999, 9.753009, 9.050719, 12.005909, 9.970784, 12.61057, 10.317732, 12.931561, 9.434133, 12.956018, 9.303395, 8.797606, 9.166465, 4.878558, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.637488, 9.158384, 9.279298, 11.525024, 9.4775095, 12.514895, 10.284993, 13.606939, 9.408581, 12.89784, 9.235064, 8.78309, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.5027113, 8.702549, 8.850396, 11.258399, 9.818504, 12.6767025, 10.400488, 13.475388, 9.471151, 12.86763, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.4030104, 8.770326, 9.159116, 11.57229, 10.052259, 12.681681, 10.472161, 13.606371, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
    elif version == 3:
        name = "double warp reduce (32 warps)"
        size_inert = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        size_op = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        caches = [false, true]
        throughput = [0.053401526, 0.07169535, 0.13477895, 0.17310828, 0.32620758, 0.26647285, 0.52654284, 0.48099294, 1.1751704, 1.274693, 2.2492318, 1.2940652, 3.5124185, 4.291191, 8.49069, NaN, 8.5058365, NaN, 8.762167, NaN, 10.001828, NaN, 10.515182, NaN, 10.761692, NaN, 10.865916, NaN, 0.12828134, 0.1535569, 0.24818721, 0.1854301, 0.52691513, 0.5276615, 1.127168, 1.1807576, 0.96136516, 1.8838382, 3.4334474, 3.4118285, 5.7065244, 3.713685, 8.67292, NaN, 8.998626, NaN, 9.527215, NaN, 9.847938, NaN, 9.87854, NaN, 9.456129, NaN, 10.812253, NaN, 0.21913472, 0.22174348, 0.41739947, 0.47185436, 0.8160548, 0.8223599, 1.6188116, 1.8306094, 2.687315, 2.9075437, 5.391646, 5.652408, 7.794004, 8.01407, 10.002877, NaN, 10.198635, NaN, 10.276109, NaN, 10.323664, NaN, 10.569078, NaN, 10.736704, NaN, 10.735872, NaN, 0.26916838, 0.30152085, 0.57224125, 0.61321646, 1.1949608, 1.2234123, 2.1809235, 2.278465, 3.2306042, 3.7784243, 6.30737, 6.1369004, 8.804231, 9.717488, 12.198443, NaN, 11.035342, NaN, 10.765338, NaN, 10.701764, NaN, 11.274829, NaN, 11.285669, NaN, 11.242099, NaN, 0.32499805, 0.33561176, 0.64675176, 0.6627156, 1.2868016, 1.1153563, 2.4283824, 2.529372, 3.9453678, 3.9447153, 6.532911, 6.693391, 10.435215, 10.43179, 12.75392, NaN, 11.0702505, NaN, 11.209313, NaN, 11.474326, NaN, 11.453139, NaN, 11.489965, NaN, 11.352855, NaN, 0.35742772, 0.39049163, 0.7526789, 0.76475036, 1.4821496, 1.5222743, 2.7888474, 2.7442286, 4.51807, 4.5206404, 7.58148, 7.647749, 11.176962, 10.956737, 13.566745, NaN, 11.465361, NaN, 11.329072, NaN, 11.404945, NaN, 11.452947, NaN, 11.499274, NaN, 11.462147, NaN, 0.36093405, 0.38090903, 0.7426445, 0.71254814, 1.4274851, 1.4791152, 2.6467428, 2.8042645, 4.3760576, 4.2639465, 7.2297354, 7.4715943, 6.8319674, 7.0855107, 8.914719, NaN, 9.675956, NaN, 11.175447, NaN, 11.410148, NaN, 11.467687, NaN, 11.517702, NaN, NaN, NaN, 0.3689776, 0.42423233, 0.81114066, 0.81625044, 1.579611, 1.6388409, 2.9923887, 3.0885236, 4.7862005, 4.702884, 7.6932483, 7.4539294, 11.275073, 11.383073, 13.181852, NaN, 11.38635, NaN, 11.499514, NaN, 11.468667, NaN, 11.52705, NaN, NaN, NaN, NaN, NaN, 0.3676861, 0.37675574, 0.7269524, 0.74581593, 1.4186515, 1.4709705, 2.7881138, 2.876065, 4.70242, 4.7632513, 7.5208683, 7.7927303, 11.248147, 11.410317, 13.165135, NaN, 11.439145, NaN, 11.547463, NaN, 11.502183, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.36652723, 0.42169988, 0.82155937, 0.85012865, 1.606689, 1.6681815, 3.0226915, 3.1333504, 4.6956477, 4.707498, 7.40319, 7.4898233, 11.184293, 11.198701, 13.142684, NaN, 11.459106, NaN, 11.5767355, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.39649203, 0.42891455, 0.8256456, 0.85123277, 1.6165957, 1.6643748, 3.0308568, 3.086, 4.629543, 4.6827793, 7.304299, 7.485193, 11.15331, 11.225952, 13.102101, NaN, 11.475384, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.39216003, 0.42983085, 0.82671195, 0.8519323, 1.611869, 1.669302, 3.0264804, 3.1049201, 4.637979, 4.694066, 7.2971206, 7.496307, 11.151141, 11.23526, 13.072195, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.4015315, 0.43095598, 0.8283816, 0.8531603, 1.6139301, 1.671738, 3.0374362, 3.1148744, 4.644622, 4.705705, 7.2721324, 7.4388127, 11.033618, 11.187621, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.4127078, 0.43231344, 0.8282116, 0.8545014, 1.6176833, 1.670799, 3.0313115, 3.1208086, 4.6174145, 4.631486, 7.1873546, 7.3575916, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.40883225, 0.4280923, 0.8228087, 0.85168284, 1.6119978, 1.667918, 3.030241, 3.1246786, 4.6350594, 4.668, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.41762036, 0.42657316, 0.8144601, 0.838059, 1.6093217, 1.6659682, 2.994751, 3.0847242, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
    elif version == 4:
        name = "double warp reduce (1 warp)"
        size_inert = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        size_op = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        caches = [false, true]
        throughput = [0.067536086, 0.07677845, 0.13163571, 0.22387561, 0.14735934, 0.27251574, 0.28115398, 0.52691513, 0.58921164, 0.8418735, 0.816502, 1.0248392, 0.87757134, 1.3224905, 0.95154285, NaN, 0.67575127, NaN, 0.65718067, NaN, 0.6729, NaN, 0.7242611, NaN, 0.8104754, NaN, 0.8331553, NaN, 0.1325726, 0.23941454, 0.31437048, 0.2695579, 0.4630566, 0.6077146, 1.127168, 1.0948685, 1.6846989, 2.085537, 0.9709178, 1.371473, 1.0214173, 1.6272085, 0.86261755, NaN, 0.84506637, NaN, 0.81614536, NaN, 1.1170744, NaN, 1.3072537, NaN, 1.5316902, NaN, 1.616851, NaN, 0.40056884, 0.2560337, 0.3444559, 0.5871222, 1.359595, 0.5644379, 2.280208, 2.2043138, 3.435426, 4.248371, 4.2620406, 5.4014177, 2.9325776, 2.6196966, 2.9235876, NaN, 2.9789288, NaN, 2.9963377, NaN, 3.0358014, NaN, 3.0319047, NaN, 3.0390546, NaN, 3.0402112, NaN, 0.4819263, 0.31597033, 1.1054274, 0.7921936, 2.1881294, 2.1705987, 4.3223095, 4.6932793, 4.58321, 9.205351, 4.908762, 4.545635, 4.909773, 3.3464606, 5.1754184, NaN, 5.233068, NaN, 5.274675, NaN, 5.265101, NaN, 5.2038174, NaN, 5.2726607, NaN, 5.2796245, NaN, 1.0192313, 1.3151951, 2.6443942, 2.245842, 4.7875214, 4.6602535, 6.847173, 8.131602, 7.0873537, 6.914692, 7.645296, 6.13769, 8.256206, 3.6363697, 8.324671, NaN, 8.27537, NaN, 8.286156, NaN, 8.429897, NaN, 8.440693, NaN, 8.411147, NaN, 8.442678, NaN, 2.8711295, 2.4773335, 4.539577, 3.5993142, 7.7509294, 8.170616, 8.212834, 9.124323, 7.067395, 9.235661, 8.010704, 7.4267917, 8.231976, 3.9634867, 8.250849, NaN, 8.234818, NaN, 8.165501, NaN, 8.272319, NaN, 8.257032, NaN, 8.2395935, NaN, 8.222476, NaN, 3.394342, 3.6815717, 6.09143, 6.1959095, 7.4158187, 7.1769595, 9.258974, 10.070478, 8.499771, 10.562346, 8.423569, 7.561942, 8.550257, 4.0832963, 8.240154, NaN, 8.154329, NaN, 8.093861, NaN, 8.170266, NaN, 8.139975, NaN, 8.089042, NaN, NaN, NaN, 4.2257814, 4.08251, 6.315724, 6.9026804, 8.741286, 9.517708, 10.054552, 11.77521, 9.314135, 11.366119, 9.009252, 7.9097147, 9.129565, 4.130226, 8.88182, NaN, 8.789474, NaN, 8.74893, NaN, 8.763306, NaN, 8.733605, NaN, NaN, NaN, NaN, NaN, 4.9168606, 5.2045093, 7.5258393, 8.355303, 9.443254, 10.624713, 10.760782, 12.604735, 9.434847, 11.752718, 9.349404, 7.8570933, 9.534895, 4.165722, 9.173639, NaN, 9.128418, NaN, 9.0721035, NaN, 9.072079, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 5.175699, 5.652408, 7.8614655, 9.133937, 9.710563, 10.934125, 10.709125, 12.586022, 9.41982, 11.919996, 9.364897, 8.051367, 9.499014, 4.1487436, 9.120615, NaN, 9.071293, NaN, 9.046142, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 5.43869, 5.9289665, 8.413536, 9.7368355, 9.5786495, 10.598736, 10.48642, 12.540508, 9.336647, 11.914041, 9.325492, 8.096717, 9.4531555, 4.167184, 9.03372, NaN, 8.960355, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 5.510498, 5.7495284, 8.135244, 9.314135, 9.16213, 10.613923, 10.668319, 12.697775, 9.347544, 11.888838, 9.301417, 8.136708, 9.387677, 4.172374, 8.962065, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 5.444356, 5.9169197, 8.042201, 9.308904, 9.682771, 11.149033, 10.571384, 12.579589, 9.315414, 11.85449, 9.264969, 8.162273, 9.310776, 4.164693, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 5.4268136, 5.7865944, 8.100349, 9.227783, 9.639312, 10.992176, 10.4887085, 12.543215, 9.320721, 11.89565, 9.235193, 8.185195, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 5.359113, 5.7197866, 8.003898, 9.234292, 9.517514, 11.002202, 10.510826, 12.532127, 9.34212, 11.900035, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 5.332892, 5.713822, 7.9666634, 9.263639, 9.588341, 11.001974, 10.46153, 12.294842, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
    elif version == 5:
        name = "double warp reduce (32 warps), group size 4"
        size_inert = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        size_op = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        caches = [false, true]
        throughput = [0.065218665, 0.0730449, 0.14273143, 0.1354651, 0.29195064, 0.3326152, 0.5318045, 0.76181805, 1.4945998, 1.2812693, 2.2340572, 1.696205, 4.1623354, 5.403866, 3.696412, NaN, 9.141816, NaN, 9.44232, NaN, 10.340175, NaN, 10.478786, NaN, 11.08698, NaN, 11.038377, NaN, 0.1185643, 0.13931526, 0.2767675, 0.25709388, 0.49803346, 0.5670153, 0.849069, 1.1542339, 2.010953, 1.8826483, 3.6211812, 4.0083823, 6.3714213, 6.9590945, 8.619616, NaN, 9.457301, NaN, 9.390255, NaN, 10.471882, NaN, 10.840287, NaN, 10.989881, NaN, 11.109663, NaN, 0.20468628, 0.22441508, 0.44349518, 0.47035566, 0.9377697, 0.8237236, 1.565248, 1.3448892, 3.4235866, 3.0931818, 5.829394, 5.835012, 8.530277, 8.585473, 11.3370695, NaN, 10.946714, NaN, 9.621898, NaN, 9.371576, NaN, 10.8430605, NaN, 10.650076, NaN, 10.9798765, NaN, 0.28889418, 0.2892306, 0.57268107, 0.56637543, 1.1641532, 1.171029, 2.176171, 1.9606992, 1.9574696, 2.1185231, 3.6129332, 3.7977026, 5.396508, 5.2835145, 7.4546757, NaN, 7.863734, NaN, 2.3359482, NaN, 10.517211, NaN, 10.851542, NaN, 11.279956, NaN, 11.214626, NaN, 0.28822362, 0.3478329, 0.70924145, 0.6955035, 1.3630149, 1.2676445, 2.6073773, 2.6782584, 4.5973506, 4.899683, 7.8042088, 8.858205, 11.515023, 12.828549, 14.276561, NaN, 11.833656, NaN, 11.595699, NaN, 11.557412, NaN, 11.533738, NaN, 11.606421, NaN, 11.432599, NaN, 0.33625665, 0.36522454, 0.7314351, 0.7030508, 1.431084, 1.419327, 2.6876178, 2.809221, 4.894151, 5.068961, 8.111545, 8.734881, 12.541745, 11.762139, 14.706416, NaN, 12.071819, NaN, 10.372439, NaN, 11.661683, NaN, 11.773212, NaN, 11.7533865, NaN, 11.541998, NaN, 0.3256017, 0.39260074, 0.7209512, 0.7846846, 1.4780148, 1.5602288, 2.854287, 2.934202, 5.176542, 5.3919506, 8.537818, 9.45121, 13.23353, 12.353294, 14.025138, NaN, 12.176055, NaN, 11.858303, NaN, 11.654001, NaN, 11.746544, NaN, 11.621158, NaN, NaN, NaN, 0.35254416, 0.37779456, 0.74484855, 0.7825727, 1.4745868, 1.543662, 2.8305662, 2.8021226, 5.1248016, 5.3506565, 8.342512, 8.939999, 12.351287, 12.4226885, 13.712684, NaN, 12.125354, NaN, 11.178068, NaN, 11.792661, NaN, 11.93774, NaN, NaN, NaN, NaN, NaN, 0.3410462, 0.35365283, 0.6537392, 0.7501804, 1.4026271, 1.6147277, 2.9223335, 3.1325014, 5.4028707, 5.7508283, 8.672033, 9.616924, 11.138775, 12.252611, 12.3481455, NaN, 11.930203, NaN, 11.897634, NaN, 11.703151, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.36627102, 0.42651668, 0.82610685, 0.8603908, 1.611617, 1.6824104, 3.0777586, 3.211917, 5.547723, 5.785367, 8.798241, 9.569339, 11.652421, 12.345848, 13.720545, NaN, 12.341766, NaN, 12.036314, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.34649462, 0.4115224, 0.8225159, 0.85289997, 1.6102835, 1.6985917, 3.082098, 3.2182033, 5.4955535, 5.784709, 8.7406845, 9.971858, 11.688841, 12.292142, 13.790061, NaN, 12.385733, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.36385897, 0.42827025, 0.8258243, 0.85139036, 1.6120425, 1.6946715, 3.0971122, 3.2369435, 5.5149794, 5.811189, 8.720404, 10.063405, 11.628413, 12.2939005, 13.815641, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.38593346, 0.4291022, 0.827625, 0.86231136, 1.6154646, 1.7017136, 3.0944426, 3.2305596, 5.519518, 5.825294, 8.726388, 10.158153, 11.6819, 12.300416, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.40691084, 0.42961156, 0.8283699, 0.862696, 1.6168518, 1.7030421, 3.09387, 3.2305305, 5.52114, 5.837353, 8.737551, 10.168543, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.41343272, 0.42725885, 0.8246664, 0.85163105, 1.6131376, 1.6930648, 3.091405, 3.2289515, 5.4874873, 5.747132, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.41195953, 0.42773825, 0.82078046, 0.8536554, 1.6117561, 1.6972816, 3.08588, 3.2222915, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
    elif version == 6:
        name = "double warp reduce (1 warp) skip second"
        size_inert = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        size_op = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
        caches = [false, true]
        throughput = [0.07577889, 0.07839416, 0.12872462, 0.08836078, 0.25136912, 0.28945532, 0.45458087, 0.41739947, 0.63517314, 0.7681011, 0.72476465, 1.0344437, 0.8311901, 1.2384094, 0.87115824, NaN, 0.63293046, NaN, 0.5997725, NaN, 0.710573, NaN, 0.7763898, NaN, 0.8371531, NaN, 0.8344821, NaN, 0.098500535, 0.080251835, 0.25001946, 0.25656268, 0.5051241, 0.947911, 1.001422, 1.1826317, 1.7187036, 2.566953, 2.251781, 3.2535288, 2.466059, 3.717159, 1.6254336, NaN, 1.5720079, NaN, 1.6026254, NaN, 1.6014547, NaN, 1.6257452, NaN, 1.6273196, NaN, 1.624935, NaN, 0.29172206, 0.2619754, 0.50172263, 0.66404456, 1.2065717, 1.0924605, 2.2441509, 2.4150991, 3.1603734, 1.7856395, 4.294283, 5.798117, 2.9672506, 2.7108426, 2.9340215, NaN, 2.9557548, NaN, 3.0304716, NaN, 2.9682894, NaN, 3.0415864, NaN, 3.0197783, NaN, 2.971049, NaN, 0.4296759, 0.62874097, 1.1854544, 0.79345906, 2.422953, 1.7256701, 4.0685763, 5.9664307, 2.5819643, 9.3717985, 5.0087934, 4.1871896, 4.9862714, 3.6275175, 4.1143894, NaN, 5.1878057, NaN, 4.9239054, NaN, 5.1487346, NaN, 4.8878174, NaN, 5.1544304, NaN, 5.2399735, NaN, 1.0948685, 1.0837208, 2.181722, 2.0191276, 4.150741, 4.938247, 9.15586, 9.2697735, 7.499798, 7.2777343, 7.3985596, 6.3232617, 8.297149, 4.362446, 7.9152956, NaN, 7.829999, NaN, 8.481157, NaN, 8.051621, NaN, 8.196996, NaN, 8.194916, NaN, 8.279114, NaN, 2.2715185, 2.5428603, 4.539577, 3.427524, 8.000623, 8.365564, 8.628975, 9.959004, 8.202945, 9.866277, 7.3939705, 7.5748553, 8.067969, 4.7157907, 8.253883, NaN, 7.782159, NaN, 7.9414954, NaN, 8.24011, NaN, 8.155725, NaN, 8.156225, NaN, 8.182811, NaN, 4.4053693, 4.992014, 7.3951173, 3.9776208, 8.475598, 9.23745, 9.932038, 11.086658, 8.677655, 11.391235, 8.716519, 8.401307, 8.55793, 4.6864176, 8.279052, NaN, 8.113227, NaN, 8.206849, NaN, 7.9838157, NaN, 8.146215, NaN, 8.065149, NaN, NaN, NaN, 5.6658406, 6.176647, 7.794004, 8.807485, 9.994492, 10.869322, 11.063506, 12.837183, 9.527692, 12.653235, 9.1767845, 8.48427, 9.049753, 4.707643, 8.793627, NaN, 8.77989, NaN, 8.783415, NaN, 8.643492, NaN, 8.636026, NaN, NaN, NaN, NaN, NaN, 6.0313325, 7.6686583, 9.003723, 10.542497, 10.57875, 11.781029, 11.3796835, 13.792382, 9.190492, 12.759038, 9.48847, 8.489556, 9.6149235, 4.817069, 9.1177, NaN, 9.082734, NaN, 9.024871, NaN, 8.895917, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 6.854063, 8.3334, 8.87717, 11.635851, 10.373354, 12.970749, 11.414756, 13.800865, 9.087264, 13.034793, 9.2993765, 8.738332, 9.421797, 4.9124603, 9.046736, NaN, 8.88299, NaN, 8.626188, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.2961087, 8.915343, 9.525789, 12.026916, 10.827366, 12.988856, 11.05421, 13.777688, 9.246294, 12.226006, 9.420227, 8.910736, 9.436992, 4.928044, 8.988687, NaN, 8.72477, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.40949, 8.599795, 9.556574, 11.977436, 10.262701, 12.916072, 10.913324, 13.945151, 9.494345, 12.990073, 9.2887335, 8.742025, 9.146022, 4.888166, 8.902954, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 7.270106, 8.194839, 9.312544, 11.91143, 10.343331, 12.784266, 10.718078, 13.401711, 9.479393, 13.008651, 9.300709, 8.958401, 9.162753, 4.9453936, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 6.9757013, 8.227001, 9.556872, 11.076998, 10.512102, 12.113851, 10.775392, 13.677224, 9.423274, 12.871551, 8.285688, 8.115978, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 6.579674, 7.894124, 8.691074, 10.812441, 9.705219, 11.808517, 10.263288, 12.70129, 8.597782, 11.195224, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 6.441637, 7.491017, 8.407776, 10.7268915, 9.76944, 11.899593, 10.195921, 12.791621, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]

    else:
        assert False, f"Unknown version '{version}'"
    # @formatter:on

    return name, size_inert, size_op, caches, throughput


def main():
    # for version in [1, 3]:
    # for version in [1, 3, 4, 6]:
    for version in [-1, 1]:
        name, size_inert, size_op, caches, throughput = data(version)

        size_inert = np.array(size_inert)
        size_op = np.array(size_op)
        throughput = np.array(throughput).reshape((len(size_inert), len(size_op), len(caches)))

        fig, axes = plt.subplots(1, len(caches))

        vmin = min(0.0, float(np.nanmin(throughput)))
        vmax = max(17.0, float(np.nanmax(throughput)))
        img = None

        for ci, cache in enumerate(caches):
            if len(caches) == 1:
                ax = axes
            else:
                ax = axes[ci]

            img = ax.matshow(throughput[:, :, ci], vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(size_op)), size_op, rotation=90)
            ax.set_yticks(np.arange(len(size_inert)), size_inert)
            ax.set_xlabel("size_op")
            ax.set_ylabel("size_inert")
            ax.set_title(f"cache={cache}")

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(img, cax=cbar_ax)

        fig.suptitle(f"LN throughput in Gel/s (version={version} {name})")

    plt.show()


if __name__ == '__main__':
    main()