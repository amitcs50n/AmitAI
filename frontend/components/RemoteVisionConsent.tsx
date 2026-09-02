export function RemoteVisionConsent({ checked, disabled, onChange }: {
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  return <label className="mb-2 flex items-start gap-2 text-xs text-[#bcb1a7]">
    <input checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} type="checkbox" />
    <span>Allow this image to be sent to the remote GPU for this message
      <span className="block text-[#948d86]">The provider can see the image and included text. Private details in pixels are not automatically removed.</span>
    </span>
  </label>;
}
