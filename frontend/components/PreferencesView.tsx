"use client";

import type { UiPreferences } from "@/lib/types";
import { SettingRow, SettingsSection, Toggle } from "@/components/SettingsControls";

interface PreferencesViewProps {
  preferences: UiPreferences;
  onChange: (patch: Partial<UiPreferences>) => void;
}

export function PreferencesView({ preferences, onChange }: PreferencesViewProps) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-3xl px-6 py-10 sm:px-10">
        <div className="mb-9">
          <h2 className="font-serif text-4xl text-[#eee8e1]">Preferences</h2>
          <p className="mt-2 text-sm leading-6 text-[#8c857f]">Lightweight display choices saved only in this browser.</p>
        </div>
        <SettingsSection title="Conversation">
          <SettingRow
            label="Enter to send"
            description="Use Shift + Enter for a newline."
            value={<Toggle checked={preferences.enterToSend} label="Enter to send" onChange={(enterToSend) => onChange({ enterToSend })} />}
          />
          <SettingRow
            label="Show timestamps"
            value={<Toggle checked={preferences.showTimestamps} label="Show timestamps" onChange={(showTimestamps) => onChange({ showTimestamps })} />}
          />
          <SettingRow
            label="Compact density"
            description="Reduce the vertical space between messages."
            value={<Toggle checked={preferences.compactMessages} label="Compact conversation density" onChange={(compactMessages) => onChange({ compactMessages })} />}
          />
          <SettingRow
            label="Wrap code"
            description="Wrap long code lines instead of scrolling horizontally."
            value={<Toggle checked={preferences.wrapCode} label="Wrap code" onChange={(wrapCode) => onChange({ wrapCode })} />}
          />
        </SettingsSection>
      </div>
    </div>
  );
}
