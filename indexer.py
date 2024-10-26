import math
import io
import requests
import json
import colorsys

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def RGB2YUV( rgb ):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb, m)
    yuv[:,1:] += 128.0
    return yuv

def YUV2RGB( yuv ):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    rgb = np.dot(yuv - np.array([0., 128., 128.]), m)
    return rgb

def calculate_hue_distance(hue1, hue2):
    """Calculate circular hue distance (mod 360)."""
    diff = abs(hue1 - hue2) % 360
    return min(diff, 360 - diff)

def calculate_chroma_distance(chr1, chr2):
    """Calculate the absolute difference in saturation."""
    return abs(chr1 - chr2)

def calculate_luminance_distance(lum1, lum2):
    """Calculate the absolute difference in luminance."""
    return abs(lum1 - lum2)

def bin_value(value, bins):
    """Assign a value to a specific bin."""
    return np.digitize(value, bins) - 1

def get_perceived_temperature(h, s, v):
    return ((h - 0.35) / 0.7 if h < 0.7 else math.tan((-h + 0.85) / 0.15) / math.tan(1) * 0.5) * s**1.7 * (0.7 + 0.3*v) * (0.6 if 0.3 < h and h < 0.55 else 1) - 0.3 * v

def get_palette_features(hsv_palette, hls_palette):
    """Convert a palette of RGB colors to their HSL representations and compute pairwise features."""
    n = len(hsv_palette)

    hue_bins = np.zeros(12, dtype=int)
    chr_bins = np.zeros(6, dtype=int)
    lum_bins = np.zeros(6, dtype=int)
    contrast_matrix = np.zeros((3, 3), dtype=int)

    hue_bin_edges = np.linspace(0, 0.5, 13)[:12]
    chr_bin_edges = np.linspace(0, 1, 7)[:6]
    lum_bin_edges = np.linspace(0, 1, 7)[:6]

    for i in range(n):
        for j in range(i + 1, n):
            h1, s1, v1 = hsv_palette[i]
            h2, s2, v2 = hsv_palette[j]
            _, l1, _ = hls_palette[i]
            _, l2, _ = hls_palette[j]

            hue_dist = calculate_hue_distance(h1, h2)
            hue_bin = bin_value(hue_dist, hue_bin_edges)
            hue_bins[hue_bin] += 1

            chr_dist = calculate_chroma_distance(s1*v1, s2*v2)
            chr_bin = bin_value(chr_dist, chr_bin_edges)
            chr_bins[chr_bin] += 1

            lum_dist = calculate_luminance_distance(l1, l2)
            lum_bin = bin_value(lum_dist, lum_bin_edges)
            lum_bins[lum_bin] += 1

            temp1 = get_perceived_temperature(h1, s1, v1)
            temp2 = get_perceived_temperature(h2, s2, v2)
            temp_dist = abs(temp1 - temp2)
            if temp_dist < 0.3:
                temp_contrast = 0
            elif temp_dist < 0.6:
                temp_contrast = 1
            else:
                temp_contrast = 2

            # Compute contrast in terms of saturation/luminance
            if chr_dist > .5 or lum_dist > .5:
                chr_lum_contrast = 2  # High contrast
            elif chr_dist > .2 or lum_dist > .2:
                chr_lum_contrast = 1  # Medium contrast
            else:
                chr_lum_contrast = 0  # Low contrast

            contrast_matrix[chr_lum_contrast, temp_contrast] += 1

    return hue_bins, chr_bins, lum_bins, contrast_matrix.reshape((-1,))

def to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return [to_list(y) for y in x]
    elif isinstance(x, dict):
        return {k: to_list(v) for k, v in x.items()}
    elif isinstance(x, np.int64) or isinstance(x, np.float64):
        return x.item()
    else:
        return x

class Indexer:
    def __init__(self, searcher):
        self.searcher = searcher

    def add(self, db, fileId, url, searchable, exists):
        # if not ((url.endswith("png") or url.endswith("webp"))):
        #   return
        if self.searcher.has(fileId):
            if exists == "skip":
                return
            else:
                db.cursor().execute('UPDATE image SET "searchable" = ? where "key" = ?', (searchable, fileId));
                db.commit();
                entry = self.searcher.data[fileId]
                if searchable:
                    self.searcher.data_searchable.append(entry)
                else:
                    # TODO: remove
                    pass
                return
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as img:
                im = np.array(img)
                im = im.reshape((-1, im.shape[-1]))
                if im.shape[1] == 4:
                    im = im[im[:,3] != 0];
                yuv = RGB2YUV(im[:,:3])
                freq, s = np.histogramdd(yuv, bins=[range(0, 255, 8),range(0, 255, 8),range(0, 255, 8)], density=True)
                freq = freq / freq.sum()
                a = freq.reshape((-1,))
                entropy = np.sum(a * np.log(a + np.where(a == 0, 1, 0)) + (1-a)*np.log(1-a))

                data_points = np.argwhere(freq > 0)
                weights = freq[freq > 0]
                k = 5
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(data_points, sample_weight=weights)
                labels = kmeans.labels_
                dominant_colors = kmeans.cluster_centers_
                distance = ((data_points[np.newaxis, :, :] - dominant_colors[:, np.newaxis, :])**2).sum(axis=2).min(axis=0)
                data_points_2 = data_points[distance>8]
                weights_2 = weights[distance>8]
                y = data_points_2[:,0]
                u = data_points_2[:,1]
                v = data_points_2[:,2]
                r2 = np.maximum(0, (u - 16)**2 + (v - 16)**2)
                k = 5
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(data_points_2, sample_weight=r2*(-np.min(np.log(weights_2))+np.log(weights_2)))
                labels = kmeans.labels_
                accent_colors = kmeans.cluster_centers_

                entry = {}
                entry["entropy"] = entropy

                dominant_colors = [YUV2RGB(yuv_color).astype(int) for yuv_color in 8 * dominant_colors]
                r = {}
                entry["dominantColors"] = r
                r["values"] = dominant_colors
                dominant_colors = np.array(r["values"])
                dominant_colors_hsv = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in dominant_colors]
                dominant_colors_hls = [colorsys.rgb_to_hls(r/255, g/255, b/255) for r, g, b in dominant_colors]
                n = len(dominant_colors)
                max_chr = max(s * v for h, s, v in dominant_colors_hsv)
                mean_lum = sum(l for h, l, s in dominant_colors_hls) / n
                mean_temp = sum(get_perceived_temperature(h, s, v) for h, s, v in dominant_colors_hsv) / n
                r["maxChroma"] = max_chr
                r["meanLuminocity"] = mean_lum
                r["meanTemperature"] = mean_temp
                r["features"] = np.array([n for ar in get_palette_features(dominant_colors_hsv, dominant_colors_hls) for n in ar])

                accent_colors = [YUV2RGB(yuv_color).astype(int) for yuv_color in 8 * accent_colors]
                r = {}
                entry["accentColors"] = r
                r["values"] = accent_colors
                accent_colors = np.array(r["values"])
                accent_colors_hsv = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in accent_colors]
                accent_colors_hls = [colorsys.rgb_to_hls(r/255, g/255, b/255) for r, g, b in accent_colors]
                max_chr = max(s * v for h, s, v in accent_colors_hsv)
                mean_lum = sum(l for h, l, s in accent_colors_hls) / n
                mean_temp = sum(get_perceived_temperature(h, s, v) for h, s, v in accent_colors_hsv) / n
                r["maxChroma"] = max_chr
                r["meanLuminocity"] = mean_lum
                r["meanTemperature"] = mean_temp
                r["features"] = np.array([n for ar in get_palette_features(accent_colors_hsv, accent_colors_hls) for n in ar])

                entry["fileId"] = fileId

                self.searcher.data[fileId] = entry
                if searchable:
                    self.searcher.data_searchable.append(entry)
                db.cursor().execute("INSERT INTO image VALUES (?, ?, ?)", (fileId, json.dumps(to_list(entry)), searchable))
                db.commit()
        return True
