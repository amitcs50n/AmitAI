import { KeyRound, ShieldCheck } from "lucide-react";

import { SettingRow, SettingsSection } from "@/components/SettingsControls";

export function SecurityView() {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-6 py-10 sm:px-10">
        <div className="mb-9">
          <div className="flex items-center gap-3">
            <ShieldCheck aria-hidden="true" className="h-7 w-7 text-[#bd8254]" />
            <h2 className="font-serif text-4xl text-[#eee8e1]">Security &amp; Keys</h2>
          </div>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[#8c857f]">
            This frontend does not store API keys or secrets. Secure credential management is intentionally deferred.
          </p>
        </div>
        <SettingsSection title="Credentials">
          <SettingRow label="API keys" description="No credentials are stored in client-side source or browser storage." value="Not configured" />
          <SettingRow label="Manage keys" description="Key management will be implemented behind an appropriate server-side boundary." value={<KeyRound aria-hidden="true" className="h-4 w-4 text-[#8c857f]" />} />
        </SettingsSection>
      </div>
    </div>
  );
}
