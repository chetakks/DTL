metadata2 = []
for k in range(10):
    metadata = [[],[],[]]
    metadata[0] = full_samp_id   #  k #.insert(k, k)
    metadata[1] = full_samp_tar  #  k #.insert(k, k)
    from random import randrange
    m = randrange(9)+1
    print m
    for l in range(m):
        metadata[2].insert(l, sample_id)
        print 'metadata', metadata
    metadata2.append(metadata)
print type(metadata2)
print metadata2
    