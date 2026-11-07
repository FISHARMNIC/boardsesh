import type { BoardName } from '@boardsesh/shared-schema';
import { sprayIdentityToken } from './spray/spray-wall-registry';

type CreateClimbKeyBoard = {
  boardName: BoardName;
  layoutId: number;
  sizeId: number;
  setIds: string;
};

/**
 * React `key` for the mounted CreateClimbScreen.
 *
 * Keyed on the authoring mode and, for a fork, its seed holds (so edit/fork/new
 * transitions, and a new seed handed to an open editor, remount with a fresh
 * undo history) AND on the board's hold-identity tuple (boardName/layoutId/
 * sizeId/setIds). The hook seeds its holds reducer once at mount and never re-sanitizes
 * the painted holds when the board changes, so a board switch under a bare-open
 * create screen (active-board fallback) must remount to drop holds that don't
 * exist on the new layout/size and to re-run the per-board draft restore.
 *
 * `angle` is deliberately EXCLUDED: WebSocket session sync updates the active
 * board's angle, and an angle-only remount would wipe an in-progress paint.
 * Angle does not affect which holds are valid.
 *
 * The spray wall VERSION is INCLUDED, for the same reason the rest of the tuple
 * is: a reset changes which holds exist under an unchanged layout and size, so a
 * create screen left open across one has to remount and drop the holds that came
 * off the wall. `sprayIdentityToken` is empty for every catalogue board, and
 * leaves out the drawn picture: a generated look landing mid-paint must not
 * remount the screen and drop its undo history.
 */
export function createClimbScreenKey(
  editClimbUuid: string | undefined,
  board: CreateClimbKeyBoard,
  forkFrames?: string,
): string {
  // A fork carries its seed holds: the climb generator hands a second climb to
  // the SAME open editor (dismissTo), and a bare 'fork' key kept the first one
  // mounted. A route's fork params never change while you paint, so this never
  // remounts mid-edit.
  const authoringIdentity =
    editClimbUuid && editClimbUuid !== 'new' ? `edit:${editClimbUuid}` : forkFrames ? `fork:${forkFrames}` : 'new';
  const spray = sprayIdentityToken(board.boardName, board.layoutId);
  return `${authoringIdentity}:${board.boardName}:${board.layoutId}:${board.sizeId}${spray}:${board.setIds}`;
}
