BEGIN {
    srand(seed)
}

{
    if (NR <= k) {
        reservoir[NR] = $0
    } else {
        j = int(rand() * NR) + 1
        if (j <= k) {
            reservoir[j] = $0
        }
    }
}

END {
    for (i = 1; i <= k; i++) {
        print reservoir[i]
    }
}
