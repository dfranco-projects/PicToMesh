<h1 align="center">📸 → 🧊<br>PicToMesh</h1>

<p align="center">
  <strong>Drop in your photos. Get a 3D model back.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.12-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/runs%20on-docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Runs on Docker">
</p>

## Try it

```bash
git clone https://github.com/dfranco-projects/PicToMesh.git
cd PicToMesh
make docker
```

Open http://localhost:3000 and drop a photo in.

The first start downloads about 5 GB of model weights, so give it a few minutes. After that they're cached.

## What it does

Turns your photos into a 3D model you can spin around in your browser and download. It all runs on your own machine. No signup, no API keys.

## One photo or a few?

<table>
<tr>
<td width="50%">

### 1 photo

Quickest. It guesses what the back of your object looks like.

Powered by [TripoSR](https://github.com/VAST-AI-Research/TripoSR).

</td>
<td width="50%">

### 2+ photos

Walk around your object and take a few overlapping shots. Slower, but the shape comes from what you actually shot.

Powered by [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3).

</td>
</tr>
</table>

## How it works

1. Drop your photos in
2. The background gets cut out
3. Your 3D model gets built while you watch
4. Spin it around and zoom in
5. Download it as GLB, OBJ or STL

## Coming next

- CLI for batch jobs, no server needed
- One-click deploy (Fly.io / Render)
- A demo GIF right here

## Contributing

PRs and issues welcome. Running without Docker, commit hooks and how it's built are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. One catch: the multi-photo model's weights (Depth Anything 3) are CC BY-NC 4.0, so any setup that includes them is non-commercial only.

---

Made by Daniel Franco · [LinkedIn](https://www.linkedin.com/in/daniel-abrantes-franco/) · [Email](mailto:daniel.franco.inbox@gmail.com)

Star ⭐ if it turned one of your photos into something you can spin around.
