import matplotlib.pyplot as plt
from matplotlib import rc


def epoch60():
    n23 = [0.94799999999999995, 0.98140000000000005, 0.9859, 0.98580000000000012, 0.98720000000000008, 0.9890000000000001, 0.98780000000000001, 0.98920000000000019, 0.99029999999999996, 0.99099999999999999, 0.99130000000000007, 0.99199999999999999, 0.9901000000000002, 0.99180000000000001, 0.99070000000000003, 0.99150000000000005, 0.99180000000000001, 0.99200000000000021, 0.99180000000000001, 0.99219999999999997, 0.99240000000000006, 0.9919, 0.9919, 0.99150000000000005, 0.99140000000000006, 0.99250000000000005, 0.99219999999999997, 0.99290000000000023, 0.99260000000000015, 0.9919, 0.99109999999999998, 0.9919, 0.99270000000000014, 0.9919, 0.99290000000000012, 0.99320000000000008, 0.99159999999999993, 0.99219999999999997, 0.99280000000000002, 0.99140000000000006, 0.99159999999999993, 0.99260000000000004, 0.99340000000000006, 0.99309999999999998, 0.99250000000000005, 0.99340000000000006, 0.99250000000000005, 0.99290000000000012, 0.99289999999999989, 0.99330000000000007, 0.99360000000000015, 0.9930000000000001, 0.9919, 0.99320000000000008, 0.99400000000000011, 0.99319999999999997, 0.99280000000000002, 0.99329999999999996, 0.99360000000000004, 0.99330000000000007]
    n24 = [0.97539999999999993, 0.97609999999999997, 0.9830000000000001, 0.98660000000000014, 0.98560000000000003, 0.98570000000000002, 0.98840000000000006, 0.98909999999999998, 0.98999999999999999, 0.98780000000000001, 0.98870000000000002, 0.98970000000000014, 0.98990000000000011, 0.9909, 0.98980000000000024, 0.98990000000000011, 0.99150000000000005, 0.9909, 0.99120000000000008, 0.99170000000000014, 0.99230000000000007, 0.99180000000000001, 0.99250000000000016, 0.99150000000000005, 0.99180000000000001, 0.99289999999999989, 0.99250000000000005, 0.99120000000000008, 0.99240000000000006, 0.99219999999999997, 0.99270000000000014, 0.99250000000000016, 0.9930000000000001, 0.9930000000000001, 0.99280000000000002, 0.99270000000000014, 0.99320000000000008, 0.99349999999999994, 0.9930000000000001, 0.99290000000000012, 0.99290000000000012, 0.99309999999999998, 0.99330000000000007, 0.99329999999999996, 0.99340000000000006, 0.99350000000000005, 0.99350000000000005, 0.99350000000000005, 0.99439999999999995, 0.99370000000000003, 0.99360000000000015, 0.99360000000000015, 0.99390000000000001, 0.99390000000000001, 0.99360000000000015, 0.99340000000000006, 0.99320000000000008, 0.99370000000000003, 0.99350000000000005, 0.99360000000000015]

    plt.axis([0, 60, 0.95, 1])
    plt.title('Влияние параметра скорости обучения на точность.')
    
    n23_, = plt.plot(n23, 'r', label='№23 (99.40%), ETA=0.6..0.1')
    n24_, = plt.plot(n24, 'b', label='№24 (99.44%), ETA=0.5..0.03')
    plt.legend(loc=4, handles=[n23_, n24_])
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.show()

def epoch40():
    n20 = [0.97149999999999992, 0.97759999999999991, 0.98340000000000005, 0.98349999999999993, 0.98650000000000004, 0.98480000000000001, 0.98610000000000009, 0.98840000000000006, 0.98710000000000009, 0.98650000000000004, 0.98799999999999999, 0.98819999999999997, 0.98880000000000012, 0.99040000000000017, 0.98719999999999997, 0.98990000000000011, 0.99010000000000009, 0.99150000000000005, 0.98970000000000002, 0.99039999999999995, 0.99099999999999999, 0.98990000000000011, 0.99120000000000008, 0.99080000000000013, 0.99150000000000005, 0.99160000000000015, 0.99210000000000009, 0.99250000000000016, 0.99180000000000001, 0.99250000000000005, 0.99170000000000003, 0.99230000000000007, 0.99260000000000004, 0.99260000000000004, 0.99199999999999999, 0.99309999999999998, 0.99210000000000009, 0.99270000000000014, 0.99250000000000005, 0.99350000000000005]
    n10 = [0.94569999999999999, 0.95909999999999995, 0.96900000000000008, 0.97340000000000004, 0.97610000000000019, 0.97840000000000005, 0.9769000000000001, 0.97939999999999994, 0.98080000000000001, 0.98020000000000007, 0.97920000000000007, 0.98120000000000007, 0.98150000000000004, 0.98180000000000012, 0.98170000000000002, 0.98219999999999996, 0.98210000000000008, 0.98199999999999998, 0.98380000000000012, 0.98409999999999997, 0.9830000000000001, 0.98319999999999996, 0.98250000000000004, 0.98340000000000005, 0.98270000000000013, 0.98219999999999996, 0.98219999999999996, 0.98439999999999994, 0.98240000000000005, 0.9819, 0.98299999999999998, 0.9830000000000001, 0.98320000000000007, 0.98089999999999999, 0.98470000000000002, 0.98230000000000006, 0.98380000000000012, 0.98260000000000003, 0.98269999999999991, 0.9839]
    n16 = [0.9728, 0.97610000000000019, 0.97999999999999998, 0.97860000000000014, 0.98080000000000012, 0.97870000000000001, 0.97349999999999992, 0.97570000000000023, 0.97620000000000007, 0.97939999999999994, 0.97620000000000007, 0.98080000000000012, 0.98210000000000008, 0.98240000000000005, 0.97230000000000005, 0.98020000000000007, 0.9819, 0.98270000000000013, 0.9778, 0.98069999999999991, 0.98070000000000013, 0.98020000000000007, 0.97860000000000003, 0.97830000000000017, 0.96650000000000003, 0.97629999999999995, 0.97960000000000003, 0.98260000000000014, 0.98499999999999999, 0.97760000000000002, 0.98280000000000001, 0.98320000000000018, 0.98050000000000015, 0.97640000000000005, 0.9839, 0.97920000000000007, 0.98040000000000005, 0.97960000000000003, 0.98099999999999998, 0.98190000000000011]

    plt.axis([0, 40, 0.9, 1])
    plt.title('Влияние завышенных параметров на точность.')

    n20_, = plt.plot(n20, 'b', label='№20 (99.35%), оптимальные параметры')
    n10_, = plt.plot(n10,'r', label='№10 (98.47%), завышенная lambda')
    n16_, = plt.plot(n16, 'g', label='№16 (98.50%), завышенная ETA')
    plt.legend(loc=4, handles=[n20_, n10_, n16_])
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.show()

if __name__ == '__main__':
    # download Verdana.ttx
    # put it in - /usr/share/matplotlib/mpl-data/fonts/ttf
    # delete MPL cache - rm -rf rm -rf ~/.cache/matplotlib/

    font = {'family': 'Verdana',
            'weight': 'normal'}
    rc('font', **font)

    epoch40()
    epoch60()