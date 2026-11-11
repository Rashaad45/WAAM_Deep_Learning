"""Extract temperatures from an Abaqus ODB of the WAAM model.  Run with Abaqus' Python:

    abaqus python extract_odb.py <job.odb> <out_prefix>

Writes
  <out_prefix>_probes.csv   time history of NT11 at every probe node (every increment)
  <out_prefix>_field.npz    nodal NT11 at every stored field frame:
                              T [n_frames, n_nodes] float32, labels [n_nodes],
                              time [n_frames] (total time, s), step [n_frames] (step name)
"""
import csv
import sys

import numpy as np
from odbAccess import openOdb


def main(odb_path, prefix):
    odb = openOdb(odb_path, readOnly=True)
    inst = list(odb.rootAssembly.instances.values())[0]
    labels = np.array([n.label for n in inst.nodes], dtype=np.int32)
    pos = {l: k for k, l in enumerate(labels)}
    steps = list(odb.steps.values())

    # ---------------- probe histories
    probe_regions = [(k, r) for k, r in steps[0].historyRegions.items() if k.startswith("Node")]
    names = [k.split(".")[-1] for k, _ in probe_regions]
    rows = []
    for st in steps:
        series = []
        for key, _ in probe_regions:
            series.append(st.historyRegions[key].historyOutputs["NT11"].data)
        for i in range(len(series[0])):
            t = st.totalTime + series[0][i][0]
            rows.append([t, st.name] + [s[i][1] for s in series])
    with open(prefix + "_probes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "step"] + ["node_" + n for n in names])
        w.writerows(rows)
    print("probe rows", len(rows))

    # ---------------- field frames
    T, times, snames = [], [], []
    for st in steps:
        for fr in st.frames:
            if fr.frameId == 0 and st.name != steps[0].name:
                continue                      # frame 0 of a step duplicates the previous step end
            fo = fr.fieldOutputs["NT11"]
            arr = np.full(len(labels), np.nan, dtype=np.float32)
            for blk in fo.bulkDataBlocks:
                idx = np.array([pos[l] for l in blk.nodeLabels])
                arr[idx] = np.asarray(blk.data, dtype=np.float32).ravel()
            T.append(arr)
            times.append(st.totalTime + fr.frameValue)
            snames.append(st.name)
        sys.stdout.write("\r%s  frames %d" % (st.name, len(T)))
        sys.stdout.flush()
    np.savez_compressed(prefix + "_field.npz", T=np.array(T), labels=labels,
                        time=np.array(times), step=np.array(snames))
    print("\nframes", len(T))
    odb.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
