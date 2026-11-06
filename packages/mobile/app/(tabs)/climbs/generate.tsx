import { useLocalSearchParams } from 'expo-router';
import { SUPPORTED_BOARDS, type BoardName } from '@boardsesh/shared-schema';
import { ClimbGeneratorScreen } from '../../../src/components/climb-generator/ClimbGeneratorScreen';
import { DevicePickerSheetHost } from '../../../src/components/ble/DevicePickerSheetHost';

type GenerateParams = {
  boardName?: string;
  layoutId?: string;
  sizeId?: string;
  setIds?: string;
  angle?: string;
};

/**
 * The experimental climb generator, opened from the climb editor's tool row on
 * the editor's own board. A modal above the editor's modal, so "Use this climb"
 * pops straight back to it.
 */
export default function GenerateClimbRoute() {
  const params = useLocalSearchParams<GenerateParams>();
  const boardName = (SUPPORTED_BOARDS as readonly string[]).includes(params.boardName ?? '')
    ? (params.boardName as BoardName)
    : 'kilter';
  return (
    <>
      <ClimbGeneratorScreen
        board={{
          boardName,
          layoutId: Number(params.layoutId),
          sizeId: Number(params.sizeId),
          setIds: params.setIds ?? '',
          angle: Number(params.angle ?? 40),
        }}
      />
      {/* Presents the Bluetooth device picker over this modal, not behind it,
          the same as the editor route below (#5868). */}
      <DevicePickerSheetHost registerExternal />
    </>
  );
}
