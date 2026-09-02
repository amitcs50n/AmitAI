import type { UploadedAsset } from "@/lib/types";
import { assetContentUrl } from "@/lib/api";

export function AssetPreview({ asset }: { asset: UploadedAsset }) {
  return (
    <figure className="min-w-0 max-w-48 rounded-xl border border-[#6f5138]/45 bg-[#171513] p-2">
      {/* Authenticated same-origin route; do not send private images through an optimizer. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img alt="Uploaded image" className="h-24 w-full rounded-lg object-contain" src={assetContentUrl(asset.id)} />
      <figcaption className="mt-1 break-all text-xs text-[#bcb1a7]">
        {asset.original_filename}
        <span className="block text-[#8f8983]">{asset.width} × {asset.height} · Stored locally</span>
      </figcaption>
    </figure>
  );
}
