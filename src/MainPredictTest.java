import org.junit.Test;
import static org.junit.Assert.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
public class MainPredictTest {

    @Test
    public void testMultipleInputsPrediction() throws Exception {
    
        String networkPath = "src/network.ser";

        String[] simulatedInputs = {
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.82965686274524e-05,-0.000240461049284586,-0.00214864174836609,-0.00116835171568631,2.97181372549027e-05,4.59558823529396e-06,0,0,0,0,0,0,0,0,0,0,0,0,0.000424632352941176,-0.00275520833333332,-0.0164043096405229,-0.0277170213199625,-0.0105818865740741,-0.0205223141339869,-0.0253208741830065,-0.00407751225490195,0.000564999388528805,0,0,0,0,0,0,0,0,0,4.84332588167164e-18,0.000167228349673168,-0.00389091435185225,-0.00818079384531673,0.0838183040577339,0.24554561914856,0.510083026960784,0.38421839256536,0.16403816040305,0.00171459694989046,-0.00619466185642668,0.000404241557734219,0,0,0,0.00015898251192369,-0.00142054738562092,-0.00250255310457517,-4.42538126361642e-05,0.000106889978213507,-0.000379791556262136,-0.0121322678376906,-0.0243612302559914,0.384323818763616,0.782284637118735,0.959642476186593,1.03784820942266,1.01526977805011,0.899124234068627,0.408965039488016,0.0036646997676406,-0.00417773011982571,0,0,0.00031388854918267,-0.00195752143899386,-0.0175354653295832,-0.0133239064856715,-0.000160409278056332,0.000233785822021124,-0.0166928332270147,0.0710476539888308,0.562982739526857,0.982583414862827,1.04672501121031,0.990549697777288,0.836288862391804,0.828658432187844,1.0175740389711,0.928540486187545,0.108205841800069,-0.0171663745193158,0,0,-0.00222562636165579,-0.00464983082630153,0.289478656045752,0.393872668164488,0.0079990298202612,-0.0268697065631804,0.112388882094765,0.724193389161221,1.02189872685185,1.01424628948802,0.666056253404138,0.351360200698435,0.0861456631263617,0.193373995778867,0.968419764433551,1.02794495506536,0.14563893644776,-0.0210282203159042,0,0,-0.0141154343681918,0.0795816008316005,0.872070993327887,0.587184351171024,-0.0339636097494561,0.159149816176471,0.71104902979903,1.06176693559368,0.972672045206971,0.482146395016339,0.0424477805010893,-0.0421263741116682,-0.0404830473856206,0.292648369417212,0.984863647194989,0.99852517361111,0.136274102156455,-0.0200217864923748,0,0,-0.0200462962962964,0.136393135216664,1.01960894948257,0.635228468818083,0.0604873195806087,0.659036951933551,1.04000884934709,0.852530330882352,0.311959456699346,-0.0225799632352944,-0.0123405671296296,-0.0174018079165144,0.0733140148420473,0.734094073393246,1.04414232706972,0.689743583197167,0.0353493538787652,-0.00918504901960787,0,0,-0.00990196078431374,0.0419187966246785,0.711455014297385,1.01903477328431,0.719560900054466,0.999011574074074,0.75334252238664,0.102800125953158,-0.0556591094771239,-0.0188325674019607,-0.0238922589869282,-0.0121903068226599,0.53728341162854,1.02709070329521,1.00975381263617,0.407450197440086,-0.0236728017610374,-0.00196861383442266,0,0,-0.000323060617178259,-0.0311982013339483,0.288284194828312,0.942797956327368,1.04940677102442,0.931020001902355,0.271653017256539,0.0805553517318218,0.114272553096082,0.107448449580802,0.160551920698979,0.680507086328292,0.992184565787507,0.957139894419306,0.517988429606077,0.0193173773320831,-0.00845818490431435,0.000260894378541441,0,0,0.000119144880174293,-0.00206983001100651,-0.016872531998911,0.347551998229848,0.908700895288671,0.999613885484749,0.924899956517603,0.924831512118736,0.925288858251633,0.925881944444444,0.945346218001089,1.03098392510157,0.884451797385621,0.28624171092048,-0.00746759259259237,-0.00862362132352936,0.000745994863641931,0,0,0,0,0.000173250173250174,-0.00217813861655776,-0.0157314814814814,0.229434283088235,0.709170870778866,0.982937779408367,0.986949669798475,1.00136943423203,1.00380450027233,0.993099247685185,0.823504086665851,0.277294015522876,-0.0182630038126361,-0.00530575980392156,0.000218290441176464,0,0,0,0,0,0,0.000261948529411771,-0.00355361519607843,-0.0233374863834424,0.0343800040849674,0.231051594581007,0.248586652369282,0.394826099537038,0.394919883578432,0.250316721132898,0.0777477647330591,-0.0173013854847494,-0.00316327954793028,0.000424632352941176,0,0,0,0,0,0,0,0,8.08823529411709e-05,-0.000314899918300612,-0.00519623161764679,-0.0167641453670862,-0.0177296262254899,-0.0257739481209148,-0.0257887561274507,-0.017849366830065,-0.00794076882312146,-0.000746272467320194,9.82945261437839e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0", // Exemplo de input 1
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5.36151960784392e-05,-0.000454963235294122,-0.0102482573070808,-0.00443167892156864,0.000390624999999999,0,0,0,0,0,0,0,0,0,0,0,0,0,4.97786271171769e-19,3.06372549019705e-05,0.00106421909041376,-0.0159481379357305,0.0380195806666389,0.00463065087145923,-0.00492493872549025,0.000459558823529423,0,0,0,0,0,0,0,0,0,0,0,0,0.000116932189542477,-0.00112353622004358,-0.0347543913398699,0.212228349673203,0.706744527332762,0.440803462009804,0.0125800823801743,-0.0052297794117647,0,0,0,0,0,0,0,0,0,0,0,5.85791951818622e-05,-0.000302933023521181,-0.0348940286440288,0.277852088513853,0.843499551587787,1.08533968348081,0.733336356718711,0.0537807094424742,-0.0120683013330072,0,0,0,0,0,0,0,0,0,3.82965686274507e-05,-0.00014910130718952,-0.00413670457788081,-0.0385537683823519,0.27391208809913,0.906345775462963,1.03962845520152,0.815036552389493,0.225292415577342,-0.0244224537037034,-0.00236938316993448,-0.00232169117647043,-0.000571674615792223,4.7487745098036e-05,0,0,0,0.000161428396722516,-0.00133670343137254,-0.00326439950980392,-0.013061785130719,-0.0294225217864923,0.00698237264413735,0.328883033769064,0.894753744553377,1.03977359068628,0.750989140795207,0.145021826124767,-0.0248819614651414,0.0044272024782137,-0.00892405364923732,-0.00934681372549006,-0.00898303846833255,0.000189440359477118,0,0,0,0.0011728017610371,-0.0104542483660134,-0.0256660198801751,0.0553730596405221,0.250057887391067,0.656171561833327,0.968844430827887,1.00585028594771,1.00134717116013,0.894123383033769,0.601322442351855,0.586647603485839,0.637855341094772,0.52627747140523,0.540905262799565,0.176778871631813,-0.0111252382897604,0,0,0,-0.0312897965839146,0.268020714188453,0.645179142837691,0.728675142973857,0.956290747549019,1.03638348755996,1.01753370098039,1.01294202750545,1.0121857128268,1.01888710171569,1.03721537714185,1.03918300653595,1.03637428513072,1.04757955473856,1.00624312363834,0.507558395499572,-0.0196825639978212,0,0,0,-0.0483803132207197,0.409783997798703,1.09457407633878,1.04750056051527,0.9634044168603,0.913861496984892,0.609705321837674,0.272162094220918,0.273934511434511,0.276556194203253,0.276556194203253,0.276556194203253,0.276556194203253,0.283021228241816,0.208616393542864,0.0809212086942304,-0.00317557376380903,0,0,0,-0.00930047694753588,0.0590748059640521,0.546876668028323,0.572750595724402,0.291545189950981,0.13070714945715,-0.0153824550653594,-0.021640114379085,-0.0206699346405229,-0.0206699346405229,-0.0206699346405229,-0.0206699346405229,-0.0206699346405229,-0.0208486519607843,-0.0190631808278867,-0.0108658432187844,0.000359136710239649,0,0,0,0.000907015612897971,-0.00960644744008713,0.0139062840413944,-0.000621136301742875,-0.0303914419934639,-0.0132786915875151,-0.000463813997821343,8.51034858387794e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000366727941176475,-0.00642095588235294,-0.00517708333333333,0.000302389705882347,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0", 
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.48412213118101e-05,-0.000206631263616565,-0.0029612608932462,-0.0152806712962963,-0.0209092456427016,-0.0131039976628212,-0.00462792755991287,0.000442538126361655,0,0,0,0,0,0,0,0,0,0,0,3.261179731768e-05,0.000950691521805645,-0.012607211283682,-0.0190782071664428,0.091778107292813,0.144750010191186,0.0713968010776128,5.18731401081482e-05,-0.0118446047857813,0.00116668704904,0,0,0,0,0,0,0,0,0,7.0210375817e-05,-0.000314627587145983,-0.023925645102116,0.10177920751634,0.481455456835512,0.87280330882353,1.04982858455882,0.825821460600873,0.493696231617647,0.0321140727124178,-0.0121968443627451,0.000201286764705877,0,0,0,0,0,0,0,0.000248161764705882,-0.00125042551742922,-0.026960018382353,0.102549104534398,0.774573427287582,1.03791084558824,0.920804279003268,0.481619570397604,0.689314676667618,1.00550010212418,0.491336226851851,-0.0106609987745097,-0.00201041666666651,0,0,0,0,0,0,3.70200163398745e-05,-0.00130829588779959,-0.0174297045206983,0.228031811683006,0.802084726128844,1.0171550585512,0.585219652096949,0.158739038671023,-0.0326051538671025,0.0459142343701161,0.729081018518518,1.00219083605664,0.242195567810458,-0.0173892463235292,7.82383411654649e-18,0,0,0,0,0,7.12316176470556e-05,-0.0206293402777778,0.134904513888888,0.79404139433551,1.04660067873303,0.617223413671024,0.0225516067538133,-0.022720520152505,-0.00341986655773409,-0.0333597794627208,0.459644080201525,1.05294272535403,0.603270578022876,-0.0388528901143789,0.000171211935917824,0,0,0,0,0,-0.00756644653703495,0.0750136392048155,0.762773395567514,1.01772670294729,0.875053068326885,0.0585958413164288,-0.0207191410867882,0.00118584647996414,0.000261302026007911,-0.0362914386150295,0.269959252238663,0.966841309635427,0.781573570176512,0.0553254555460438,-0.0103795011462861,0,0,0,0,6.11471199706458e-05,-0.0263469669117651,0.393761863425926,1.00347734545207,1.01562406386166,0.639340766105472,0.00761526416122008,0.00495179738562095,-0.000748161764705889,0,-0.0486380498145208,0.41608370778867,1.06422261369826,0.706817282815905,0.00492529616013069,-0.00487954017365786,0,0,0,0,-0.000130447189270724,-0.0361192980664493,0.606284688180828,1.01318120234205,0.998040679466231,0.817352907205848,0.563657407407407,0.0863035300925925,-0.0138465073529412,0.000245098039215687,-0.0518910765969594,0.418426947167755,1.07421852873094,0.658688538262528,-0.0444900428921567,0.000428029839794548,0,0,0,0,-0.0117361705597001,0.0677303751361652,0.786076031454249,1.01834904343682,1.00637028526688,0.857418504477328,0.211641271786492,-0.00711853213507628,-0.00133517156862744,-0.00167568763616558,-0.0342198137051083,0.475584660947711,1.05888066789216,0.547971200980393,-0.0342112438725488,0,0,0,0,0,-0.0124597448126861,0.0750406113834416,0.781448631535947,1.02901363357843,1.01223333673747,0.41648854170913,-0.0303808210784313,-0.00031668709150321,0.000117647058823524,-0.00823716639433554,0.0238103318250373,0.661511437908496,1.02202192265795,0.359870455473857,-0.0238535539215683,0,0,0,0,0,-0.000796921134453238,-0.00401451224980658,0.167613937466878,0.519014750044162,0.993932065549712,0.592277004335216,-0.0252113821966767,0.000398730178141951,0.000224206106559047,-0.0191937412525649,0.10775500573715,0.906683244330302,0.910405914964738,0.107036963434023,-0.00975097835391918,0,0,0,0,0,0.00017528841058253,-0.0008115468409586,-0.0464116625816993,0.203796364379085,0.9748269165305,0.883146138219668,0.0776443355119821,-0.0202272433278863,-0.0045244757625272,-0.0167565870098036,0.442918908727732,1.04060127314815,0.610519284449891,-0.00686662581699325,-0.00266697303921548,0,0,0,0,0,0,0,-0.0149473039215686,0.0808073257080605,0.754450027233114,1.03631381381381,0.610994842728758,0.0788394097222225,-0.0193616898148153,0.291164743327888,0.913867402470344,1.00889329724946,0.308834967320261,-0.0269681883169934,-0.000564185049019569,0,0,0,0,0,0,0,0.000265012254901957,-0.0197969430827893,0.173911611519608,0.836873649667768,1.03927913943355,0.916393348311547,0.742598226443354,0.971176640795207,1.03744106097047,0.667968784041395,0.0530899373638345,-0.0120711124727669,2.76756535947692e-05,0,0,0,0,0,0,0,6.5257352941172e-05,-0.00080324074074083,-0.00911684708605652,0.190738402429578,0.637871340550109,0.780189031862744,0.891432291666666,0.916645543981481,0.714606518282988,0.127249540441177,-0.0207360940904138,0.000758374183006529,2.45098039215653e-06,0,0,0,0,0,0,0,0,0.000142676613264865,-0.00136205209734621,-0.0228289530723189,0.0189790469202231,0.0640995067465652,0.0970150014267659,0.108599578084872,0.0512419475853494,-0.0164197851697853,0.00130319799437447,0,0,0,0,0,0,0,0,0,0,0,7.99972766884536e-05,-0.000143186172597933,-0.00739991830065363,-0.0122269880174292,-0.0157809095860567,-0.01705661083878,-0.0107041430570843,-6.04234749455375e-05,2.9786220043573e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000174632352941177,-0.000380640821817283,-0.0162594975490196,-0.0154099264705882,7.0465686274516e-05,6.98529411764658e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,1.72410349129822e-17,-0.000729847494553476,-0.0103015062573893,0.10046490332244,0.113099247685185,-0.000179908769063219,-0.000520016339869247,0,0,2.42670807196254e-18,-6.78752321837386e-18,-1.97460406171807e-16,-1.62965497133937e-16,5.85935923358473e-18,0,0,0,0,5.10620915032699e-06,0.000485430283224402,-0.0043157679738563,-0.0305974264705884,0.142068539127362,0.694187108523964,0.456141118600217,-0.005929670479303,-0.00198304738562078,0,0,0.000174632352941178,-0.000466026688453165,-0.0144653799019609,-0.0148334420393245,0.000430623638344227,0,0,0,6.83423943788393e-05,0.000509865068688618,-0.0235434756022993,-0.0134045357574778,0.240460284945578,0.856558467198067,0.690416191077955,0.109650493253434,-0.0109871183400596,-0.000267212914271719,0,0,-0.0033860217683747,0.00602851154321741,0.314760744907804,0.293115800441226,-0.00955253896430361,0,0,0,0.000733357792181301,-0.0165180078976034,0.125280620234206,0.48002987132353,0.950262084694989,0.639106215944451,0.0177473958333332,-0.0206297147331151,-0.00528167551742881,-0.00551404207516302,-0.00531373568138238,-0.00325454452614357,-0.015510365604575,0.058183823529412,0.760296194172114,0.661782965091789,-0.0210656658496731,0,0,0,-0.0239484733602383,0.18619759327342,0.812591435185185,0.982175891884532,0.706385723039216,0.322652646990882,0.166518314270153,0.0283440904139437,0.0297849434912857,0.0312801776960788,0.0282253237400299,0.00288080405773442,-0.0307959729030499,0.186910947712418,0.927434027777778,0.526469721984428,-0.0205809163943354,0,0,0,-0.0301740654681834,0.244298628131808,0.861879544526144,1.0044910130719,0.971567095588236,1.00513418395771,0.974260076252724,0.913864055691722,0.912490196078432,0.916747549019609,0.891618615221557,0.66777440767974,0.568034024373639,0.547438265931373,1.04336645561002,0.260319306863424,-0.020853587962963,0,0,0,0.000480208715502839,-0.0147914964596951,0.155575163398692,0.367526875680827,0.454836277913942,0.614996042422512,0.628019080201524,0.630886182598038,0.629797249455337,0.629400122549019,0.632163011427716,0.63676225490196,0.737320176334422,1.00184170751634,0.957079469635076,0.164889697389697,-0.0183391203703704,0,0,0,0.000126921589560702,9.00900900901096e-05,-0.0209807998043294,-0.0459521116138768,-0.0261609290285765,0.0120826200562041,0.0147330928213277,0.0140528311116542,0.0141710488769308,0.0141873547755897,0.0140465994446668,0.0115699013493128,0.0413091258679491,0.246848358539535,0.409923718967836,0.0623126527247842,-0.00858981153098802,0,0,0,0,0,0,0.000199993191721139,-0.00213694852941177,-0.00662070441482208,-0.00692572167755994,-0.00686274509803924,-0.00686274509803924,-0.00686274509803924,-0.00686274509803924,-0.00663909313725493,-0.00998212826797389,-0.0192560253267974,-0.0184349468954248,-0.0025931474460886,0.000333605664488016,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4.12071078431396e-05,0.000441227532679742,-0.0115596915849675,-0.0258818933823533,-0.0369903897109784,-0.0345329350490201,-0.012910488153595,0.000117698120915034,-0.000110089869281054,-7.64338999633033e-07,1.5318627450981e-06,0,0,0,0,0,0,0,0,0,-0.000602583741830086,-0.00735152845860567,0.17613407203159,0.398075725081699,0.598202903129374,0.553318116830065,0.174689627587146,-0.0278372821350764,-0.0130633340141611,0.000792721454486171,4.85089869281047e-05,0,0,0,0,0,0,0,0,0,-0.00263934844771251,-0.0032859647331155,0.552581239787582,1.00195538875272,1.017999181308,1.02339105051743,0.798562465958606,0.1955913500817,0.116143433415033,-0.00594023888141546,-0.000851290168845407,-0.0140990944989113,-0.0229902811819179,-0.000295309095860686,0.000163058986588417,0,0,0,0,0,5.45343137254907e-05,-0.0211904616013072,0.124144233387799,0.827171347358387,1.01837853124618,0.988045615468409,0.631608183551198,0.790327597358388,0.463151807598038,-0.0200485610044437,0.0010289862472767,0.146396377995642,0.243030450027233,-0.000700061274509937,-0.00133300721536016,0,0,0,0.00017936488524724,-0.00175818945538893,-0.00740802454037756,-0.0240547844224315,0.27682364491188,0.93632670566494,1.01198266591443,0.940431613740436,0.268464629788159,0.104672251436957,0.0735434756022989,-0.0150733117102203,0.0845297276914923,0.754737237237238,0.862509206038618,0.168032744282744,-0.022225922971918,0,0,0,-0.000788228485838795,-0.000190745043685744,0.0163435627723318,0.0303243293845322,0.769850098720044,1.0144645458878,1.02431861725979,0.727142803649237,0.0443425074891071,-0.0355543130446621,-0.011508629493464,-0.000253352900411746,-0.0349484953703701,0.703360787717865,0.924332363153595,0.199128148828976,-0.0257021727609965,0,0,0,-0.0160903458605666,0.266064673270556,0.723164692265796,0.584061291530501,0.990199499591503,0.993911917892157,1.01045248868778,0.854855119825707,0.301226970996732,0.00734861791939015,-0.0111863425925924,-0.0283648241001184,0.110879357298475,0.833885008169935,0.906349145561002,0.194871374591503,-0.0252415311238843,0,0,0,-0.0214421636710243,0.768371839033604,1.04419163602941,1.00513391884532,0.992283394607843,0.992083333333334,0.991674208144796,1.00089259940087,0.94073696214597,0.756072882625273,0.377020016339869,-0.0160676456999999,0.328999131944443,0.99824297045207,0.890280841503268,0.194766544117647,-0.0252415311238843,0,0,0,-0.019596779684096,0.286996368540485,0.991522790713507,1.01815379901961,1.01292146650327,1.01281576797386,1.01275586672645,1.01341436887255,1.01442859817538,1.02538614855664,0.929985447303922,0.66762380593263,0.732245455473856,1.01786160471133,0.884321895424837,0.194779411764705,-0.0252415311238843,0,0,0,-0.00422322775263952,0.00856420714559364,0.240437881320234,0.417207326104385,0.309280621118856,0.289472351310586,0.513214752838087,0.471175825146414,0.716898719986956,0.995768619298031,1.01893604011251,1.04614165910485,1.0166696221108,1.00901568423627,0.882667967741497,0.194779411764705,-0.0252415311238843,0,0,0,0.000377859477124185,-0.00276673732556085,-0.0195967796840959,-0.0208018450435729,-0.0206699346405229,-0.0206699346405229,-0.0206699346405228,-0.0293724468954245,0.104566533905229,0.381559878812637,0.68282584422658,0.880460794504912,0.913523335375817,1.01301848447712,0.883211907679739,0.194779411764705,-0.0252415311238843,0,0,0,0,0,0,0,0,0,0,0.000763378267973834,-0.0117526722494553,-0.0395414794389979,0.0638072065631809,0.17348205671735,0.640745829929193,1.03007048270697,0.884032679738562,0.194779411764705,-0.0252415311238843,0,0,0,0,0,0,0,0,0,0,0,7.72058823529411e-05,0.000428615196078449,-0.0143311206427015,-0.0537699237699241,0.343582516339869,0.989558738425926,0.890763378267974,0.194779411764705,-0.0252415311238843,0,0,0,0,0,0,0,0,0,0,0,0,0,8.0678104575161e-05,-0.0394960458195756,0.319330507897603,1.00162579997277,0.892331869553377,0.194665543300653,-0.0252415311238843,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.0193962224046609,0.14296692620222,0.798792718057424,0.937421137200549,0.205554910113733,-0.0265009838202943,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00036484448249154,-0.0113865059912852,0.110721081835513,0.484762646377996,0.114837622549019,-0.0144612938730587,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.05735599853249e-05,0.00048764297385619,-0.0114681202342048,-0.0441572201797384,-0.0104294321895424,0.00131466307936897,0,0",
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.29631841451003e-19,2.25559404124725e-18,-2.3869938346315e-17,-3.20885475054124e-16,-5.8560925111802e-16,-5.77872822820225e-16,-4.47375040668341e-16,-3.96614574976885e-16,-3.69715197091882e-16,-1.49506995382256e-16,2.20374130466679e-18,7.15567764809425e-19,0,0,0,0,0,0,1.73611111111118e-05,0.000245438453159042,-0.00514756944444445,-0.0144017224945534,-0.0202792724851549,-0.0197528594771244,-0.0178465413943359,-0.0181478077342051,-0.0200268927015253,-0.020841316429552,-0.0207858455882355,-0.0206973379629631,-0.0148388139978214,-0.00464596949891068,0.000483062247768136,6.80827886710244e-06,0,0,0,0.000312422374303265,-0.00185082140964494,-0.0239826138355552,0.00198374845433625,0.0949254005136354,0.194031108840941,0.539382414088297,0.823938061364532,0.817175835337601,0.682424958895547,0.630559440636486,0.60042395336513,0.34832558803147,0.0937407430054487,0.00518680445151007,-0.00695933042759512,0.000470832823774006,0,0,0.000115910947712414,-0.00203288695935755,-0.00520436751089298,0.159489328022876,0.563324329384532,0.752735140931372,0.776080214830214,0.605146854575163,0.505383680555555,0.522140114379084,0.628235804738561,0.674014002690473,0.666781198937908,0.701629918981481,0.801051079112201,0.440302372685185,0.00316961191961174,-0.00434317129629633,0,0,-0.00127331835511983,-0.0236183298683302,0.336657662717865,0.817237642973856,0.547342660675381,0.191426351443355,0.0882724070224069,-0.0196995506535947,-0.0468977566721131,-0.0419027607570805,-0.00955865332243997,0.00397167529520479,-0.00571124387254883,-0.0116918232570805,0.419473481753812,0.946574669798474,0.129368655986303,-0.0187523828976035,0,0,-0.0101201661220044,0.0518175981411274,0.67667379833878,0.419785283905228,-0.0195985498366021,-0.0298450776143792,-0.015823040234805,-0.00336029411764717,0.00030422794117607,-1.11996187368669e-05,-0.0138560389433558,-0.0298459941842303,-0.00777425449346458,0.115935934095861,0.738046262254903,0.922817436002178,0.103864497982145,-0.0166081154684096,0,0,-0.0151833129084968,0.103951887407769,0.780305385348584,0.230459882216776,-0.0471242510893243,-0.0210204248366013,-0.0201653689888984,-0.0207516339869281,-0.0212755310457515,-0.0251448461328976,0.0765541598583883,0.282169329963448,0.624399782135077,0.851276450163399,0.979849485974945,0.387025377859476,-0.0169015225632876,-0.00233404820261438,0,0,-0.00393243922655688,-0.00123463789845138,0.430835490467843,0.873378904923023,0.668015758971642,0.376716280760399,0.247050229880118,0.311594716888835,0.640101334366041,0.755890454934573,0.886960495563437,0.92151891310507,0.923114664438194,0.650619216501569,0.194895404454228,0.0124779870368104,-0.00451410667871476,0.00015133912192736,0,0,0.000335307734204794,-0.00550884595002246,-0.00810631127451002,0.346360736655774,0.713210852396515,0.838860668572985,0.782110153139565,0.77070751633987,0.60701130174292,0.618057649101308,0.614325180419391,0.32813630372454,0.0806855085784318,-0.00669160198801728,-0.0207024441721133,-0.00525854438997822,0.000461151196445321,1.53186274509805e-05,0,0,0,0.000285353226529699,-0.00145804398148147,-0.0283741149237472,0.0266711941721134,0.121479830473856,0.0954159872542225,0.0742557189542482,-0.040924019607843,-0.034302372685185,-0.0287962622549019,-0.0301719932602285,-0.00897611996187362,-0.00126676538671023,0.000214460784313725,0,0,0,0,0,0,0,6.34191176470603e-05,-0.000259191176470588,-0.00897794117647059,-0.0196344975490196,-0.0165100281276752,-0.0142794117647059,-0.00124019607843137,-0.0020030637254902,-0.00255606617647058,4.95291671762278e-05,1.83823529411765e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.02823976638923e-17,0.000314031862744951,-0.00127399918300699,-0.0199657032952077,-0.0458964337640816,-0.0523437500000007,-0.0526756535947721,-0.0467056440631817,-0.011202597358388,0.00142676613264853,0,0,0,0,0,0,0,0,8.16993464052292e-05,-0.000105988341282454,-0.00761914488017432,-0.0221359272875818,-0.00467422385620977,0.217529871323529,0.521706293397469,0.597192946623093,0.591131535947712,0.511331358932461,0.119787326388888,-0.015335697688639,0,0,0,0,0,0,0,1.25429721550983e-17,0.000224206106559063,-0.019766410374213,0.00962256940198118,0.293400204503146,0.706815678800973,0.889544046308753,0.777806637975404,0.737848776378189,0.946940792602558,1.00765923729159,0.313149348443465,-0.0379731496767088,0,0,0,0,0,0.000198529411764697,-0.00187025122549002,-0.0169172283496729,-0.0526644369553371,0.161307257483728,0.674923406862746,0.942836941721132,0.672845298883442,0.270568201933551,0.0248795741442795,-0.0412634803921568,0.317504799836602,1.01968258101852,0.407946963507624,-0.0474285597815014,0,0,0,0.000174632352941177,-0.00071817292405527,-0.0145626531862745,-0.0142591911764704,0.25319687840414,0.713273624727668,0.806729750112103,0.482382812499999,0.227672402641612,0.0408042790032678,-0.0313194955065358,-0.00390940714470125,-0.0201532713779953,0.314062091503269,1.00037057461874,0.369770782271241,-0.0434242387183568,0,0,0,-0.000508578431372533,-0.01175356351827,0.0350889501633979,0.531626906318083,0.809882846541395,0.569971405228758,0.288645353498294,0.0331222766884529,-0.0399717116013072,-0.0100526960784314,0.00138660811546805,-0.033594108814698,0.144219039351851,0.817539232706972,0.882870421432462,0.219150258714596,-0.0275940646528884,0,0,0,-0.0138696555010894,0.211562733548028,0.738054483251634,0.956815070125272,0.289419270833332,-0.0532597528594764,-0.0331442188795127,-0.0181088303376906,-0.0131090005446624,-0.0157272943899783,-0.02911628540305,0.0772015171279878,0.608988272739651,0.959369536356208,0.309634803921569,0.00283426947167781,-0.00241449594390773,0,0,0,-0.0190230388759804,0.703349345149517,0.747270358593889,0.193743562567092,-0.0184521965404323,0.136141769597652,0.0881688246981138,0.0767516781487366,0.0720703905262725,0.11282387591211,0.559034402049108,0.880855069799671,0.862577979563274,0.359015616295028,-0.0222396831220362,-0.00490985895397651,0.000419817565470013,0,0,0,-0.00273948120915038,0.100062505944859,0.0691602839052291,-0.0493270186546841,0.13643004493464,0.850531760620914,0.817612510700745,0.838630361519607,0.844149867238562,0.821239021650327,0.917355613425926,0.493597522862229,0.080516067538127,-0.0291697303921568,-0.00237983387799566,0.00021088643790849,0,0,0,0,0.000276586328976038,-0.00941241014770424,-0.00788271037581698,-0.00857140182461873,0.0633656045751631,0.312787717864923,0.614038478523772,0.656184521377995,0.430047249455337,0.245489413126361,0.164865485430283,0.030129343144049,-0.0136611519607843,0.000893586601307184,0,0,0,0,0,0,0,0,0,0.000716605392156861,-0.00484070329520698,-0.0260820738017429,-0.00230663921840374,0.00379212622549038,-0.0216568967864923,-0.0273654854302832,-0.0251126429738562,-0.00742417757123639,0.000514705882352941,0,0,0,0,0,0,0,0,0,0,1.74632352941165e-05,-0.000118515114379075,-0.000555249183006497,-0.00286943051648915,-0.00332215073529389,-0.00125081699346397,-0.000197661356209137,1.80759803921555e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6.20939215598928e-19,-3.89932579084577e-18,-3.91851130338053e-17,-1.35439347947994e-17,8.71125974550628e-19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000485100485100491,-0.00374217047930283,-0.017770288671024,-0.0102306304466231,-0.00625765931372551,-0.00266193795605561,0.000132761437908498,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.00272962777548812,0.0179511978041387,0.157145907219436,0.0610657773157769,0.00751431861725951,-0.0115348130512619,-0.0247615262321147,-0.00477151359504304,0.00069707716766541,0,0,1.34523663935421e-05,0.000832008479067309,-0.00889527536586367,-0.0282491541315073,-0.00814291493024531,0.000554400554400559,0,0,0,-0.0352774041009338,0.290538109340958,0.954599502995643,0.712154564950981,0.603060576661221,0.374585405541289,0.176013378267974,0.00884460103485904,-0.0412068014705876,-0.0348919526143784,-0.0347132709632703,-0.0348527369281039,-0.0423238017429187,0.0463669662309374,0.222216979847495,0.0445248529072064,-0.00433329929193902,0,0,0,-0.0489107659695899,0.418633186955337,1.02834640522876,1.02332497617102,1.02368949142157,0.946395241395241,0.850690410539216,0.615898539624183,0.551189133986928,0.560513208061003,0.557569215142745,0.557678513071896,0.547646956699347,0.668026450163399,0.924215192674292,0.592586132512604,-0.0186710239651415,0,0,0,-0.00195956137132607,0.00298260484749405,0.227930640659041,0.840424291938998,1.03353919866558,1.03587982212982,1.04282021037582,1.04553860294118,1.03788058278867,1.01086109409041,1.03197449145979,1.04582701865469,1.04517737268519,1.04813007216776,1.01951916530501,0.782551363580776,-0.0201572712418298,0,0,0,0.000309404427051468,-0.00301741217320245,-0.0264508612472763,0.246547300517429,0.605006127450979,0.598438693218104,0.597197252859476,0.582623263888888,0.71488929738562,1.03600907203159,0.742841795415324,0.589959405637254,0.598490655637254,0.609101068899781,0.4825865332244,0.190625823787588,-0.00798781318082784,0,0,0,0,0.000114956585544826,-0.00166085769026945,-0.0108265561942035,0.00861797317679629,0.00861297229158264,0.00851167909991402,0.00160602910602871,0.0599790061554765,0.2038873771962,0.0747914450984033,0.00469864661041092,0.00851167909991399,0.00947856793444989,-0.00233352696588028,-0.0210656786173482,0.000209428885899473,0,0,0,0,0,0.000217524509803922,-0.00159807325708062,-0.00628846677559914,-0.00622953270012095,-0.00620915032679741,-0.0056668709150327,-0.0102546296296297,-0.0215068423202615,-0.0112814737814738,-0.00591128812636168,-0.00620915032679741,-0.0063836124727669,-0.00434963916122005,-0.00024305980188333,5.95724400871463e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.87782688021186e-18,-2.79175133748854e-17,-2.9132422992964e-16,-4.16258213446446e-16,-4.226008811497e-16,-4.28013228829222e-16,-3.10271219876116e-16,-1.12313027433138e-16,-8.23421456896694e-18,1.61444196055721e-18,0,0,0,0,0,0,0,0,0,0.000599241775712371,-0.00488562091503271,-0.0164256535947713,-0.0206348720043575,-0.0206587009803924,-0.0206040316334437,-0.0205834694989109,-0.0208920547385623,-0.019274748093682,-0.00795649509803923,-4.53507806448921e-05,7.99972766884536e-05,0,0,0,0,0,9.53895071542139e-05,0.000563776446129424,-0.0163132363132364,-0.0480922988444089,-0.00462063307651554,0.134158955335426,0.469533838136779,0.60999199991847,0.616689341527302,0.622028385851915,0.502984115337056,0.272685632465044,0.0412661190602364,-0.00635518162728641,3.21022379845973e-05,0,0,3.57434640522852e-06,0.000127033141739017,-0.00106857638888879,-0.0090325776143787,-0.0403079554738558,0.110193287037038,0.344286922669276,0.54525553172658,0.915860345179739,1.03873206018518,1.05152828839869,1.05332828869593,1.05762161288126,0.96796580541939,0.831449550653595,0.58190727124183,0.0438601157718803,-0.0088967184095861,0,0,0.000580916394335513,-0.00551806897395134,-0.0222291326252724,0.0801708877995645,0.564472971132898,0.842505650871459,0.991815271153506,0.929651177832243,0.66627253540305,0.538893518518518,0.34251531862745,0.268620622885328,0.270471064814815,0.180284722222222,0.523130446623093,0.907689133986928,0.101776323835147,-0.0162152777777778,0,0,-0.00525939542483663,0.0175828543475601,0.40457097630719,0.847526416122005,0.996968613834424,0.876320942265795,0.345659522644815,0.0958937397875804,-0.00103067129629723,-0.00712132352941259,-0.030399475762528,-0.0685365795659922,-0.0279530569172119,0.289452171840959,0.915268518518519,0.962961907679739,0.1187496942644,-0.0182531658496733,0,0,-0.0152476511437909,0.0899759997554111,0.880500306372549,0.985956035539215,0.57456442333878,0.903848447712419,0.227095223051106,-0.0447475490196073,-0.02237433619281,-0.0247512595315902,-0.0127760757080606,0.226058898264781,0.601915049700436,0.97392301538671,0.925591503267973,0.560627706290849,0.0447825200766373,-0.00883442265795208,0,0,-0.0030999891294009,-0.00969826375231818,0.426277057600587,0.688324093324093,0.111173107496636,0.71681795483266,0.966470394157811,0.874661007161008,0.87866097998451,0.881360812169636,0.891442715339775,0.94858736592869,0.994943370972783,0.783759953392306,0.197755815770522,0.000899932738167818,-0.00678969484154768,0.00043006807712691,0,0,0.000260416666666667,-0.00138956830133301,-0.01632914624183,-0.0204689202069715,-0.0136792790032683,0.125826950571895,0.530443316619788,0.602779071350763,0.629516935593683,0.761966520288672,0.677791479438999,0.488202189746308,0.343944376361657,0.0293019812091505,-0.0254896854575164,-0.00588048066448803,0.000124332477273659,4.76579520697171e-05,0,0,0,0,0,7.65931372549017e-05,0.00048296228213509,-0.014623587282135,-0.0121884384384384,-0.00348168572984738,0.00851164215686284,0.067886761301743,0.0160946861383442,-0.033566728493199,-0.0333035471132897,-0.00441006263616556,0.000735294117647056,0,0,0,0,0,0,0,0,0,1.77336359104955e-18,0.000171568627450982,-0.00353338632750398,-0.00496813725490196,-0.00641513480392157,-0.0135606617647059,-0.00760723039215686,-0.00101076189311483,0.000153186274509804,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9.78353919530332e-06,0.000407647466470999,-0.00525416819534471,-0.0120349761526233,-0.00107517231156708,0.00019159430924137,0,0,0,0,0,0,2.29779411764698e-05,-0.000238970588235277,-0.00103002323590552,-0.00103048406862738,-0.00244403594771225,-0.0036181066176468,-0.00726434844771203,-0.0135096408625816,-0.0190277777777773,-0.0311683176742914,0.0144528356481485,0.082253829656863,0.00646151807916514,-0.0012258306100218,0,0,0.000180759803921569,-0.000782377400024452,-0.0128039215686274,-0.0180575980392157,-0.0228501838235294,-0.0300049019607843,-0.0239128042069218,-0.0211273999183006,-0.00293072576252706,0.0153980290032681,0.0830971030773425,0.203586210645034,0.313860174972767,0.465163126361656,0.660287224264706,0.64821616285403,0.0742014695691164,-0.0118469158496732,0,0,-0.000988221677559918,0.00367912029676732,0.0748328227124185,0.113334184368192,0.166004936002179,0.290784381808279,0.423436807848574,0.409026484204794,0.558786168981483,0.661868293845317,0.752764110157953,0.732431396328455,0.591476222086057,0.578631093409586,0.301722528594771,0.0646612030228755,0.00427989075047889,-0.00105255991285403,0,0,-0.00920632489106757,0.0466666157107331,0.597021071623094,0.753113596132898,0.634943457244008,0.712589460784313,0.593560579810579,0.443589324618735,0.397928172657951,0.331906692538126,0.175585818355119,0.0449273708097233,-0.0350522705610022,-0.0332349707244009,-0.0185890012254899,-0.00572824754901926,-0.000478476213770285,9.70179738562029e-05,0,0,-0.00201836451836452,0.0109565977488198,0.125026395173454,0.147385654885655,0.0844242998654762,0.0539407280583749,-0.00313078339249348,-0.0369024907260206,-0.0448755656108601,-0.0375781664016962,-0.0221507480331012,-0.00873765648132783,0.000179364885247238,2.03823733235499e-05,0,0,0,0,0,0,0.000243395969498912,-0.00129275202804614,-0.0153073937908497,-0.0184698393246188,-0.012332175925926,-0.0105364923747277,-0.00491232182408654,-0.000874863834422658,9.36138344226579e-05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"            
        };

        int[] expectedResults = {
            0,
            1, 
            0,
            1,
            1,
            0,
            0,
            1,
            0,
            1
        };

        // Carregar a rede neural treinada
        Network network = NetworkUtils.loadNetwork(networkPath);
  // Armazenar erros acumulados
  List<String> errors = new ArrayList<>();

  // Processar cada input
  for (int i = 0; i < simulatedInputs.length; i++) {
      try {
          // Converter os valores para double[]
          String[] values = simulatedInputs[i].split(",");
          double[] inputData = Arrays.stream(values).mapToDouble(Double::parseDouble).toArray();

          // Normalizar o input
          double[] normalizedInput = DataLoader.normalizeInput(inputData);

          // Fazer a predição
          int predictedLabel = network.predict(normalizedInput);

          // Logar a predição
          System.out.println("Input " + (i + 1) + ": Predição = " + predictedLabel + ", Esperado = " + expectedResults[i]);

          // Verificar a predição
          if (predictedLabel != expectedResults[i]) {
              errors.add("Input " + (i + 1) + ": Predição = " + predictedLabel + ", Esperado = " + expectedResults[i]);
          }

      } catch (Exception e) {
          errors.add("Erro ao processar o input " + (i + 1) + ": " + e.getMessage());
      }
  }

  // Reportar resultados ao final
  if (!errors.isEmpty()) {
      fail("Erros detectados: \n" + String.join("\n", errors));
  }
}
}
