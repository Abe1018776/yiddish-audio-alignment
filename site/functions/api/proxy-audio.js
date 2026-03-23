// Cloudflare Pages Function — proxies audio downloads to avoid User-Agent blocks

export async function onRequestGet(context) {
  const url = new URL(context.request.url);
  const audioUrl = url.searchParams.get('url');

  if (!audioUrl) {
    return new Response('Missing url parameter', { status: 400 });
  }

  if (!audioUrl.includes('.laravel.cloud/')) {
    return new Response('Forbidden origin', { status: 403 });
  }

  const resp = await fetch(audioUrl, {
    headers: { 'User-Agent': 'YiddishAligner/1.0' },
  });

  return new Response(resp.body, {
    status: resp.status,
    headers: {
      'Content-Type': resp.headers.get('Content-Type') || 'application/octet-stream',
      'Content-Length': resp.headers.get('Content-Length') || '',
    },
  });
}
